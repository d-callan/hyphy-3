#pragma once

#include "hyphy/core/types.hpp"
#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <stack>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace hyphy::core {

struct TreeNode {
    int32_t id = INVALID_INDEX;
    std::string name;
    int32_t parent_id = INVALID_INDEX;
    std::vector<int32_t> children;
    Scalar branch_length = 0.0;
    bool is_leaf = false;
    std::string model_tag;
};

struct WheatstoneBridge {
    int32_t hub_node_id = INVALID_INDEX;       // Node u (collapsed 3-child hub: A, B, C)
    int32_t bridge_node_id = INVALID_INDEX;    // Original node v in backbone tree
    std::array<int32_t, 3> children{INVALID_INDEX, INVALID_INDEX, INVALID_INDEX}; // {id_A, id_B, id_C}
    std::array<Scalar, 3> weights{1.0, 0.0, 0.0};        // {w_1, w_2, w_3} summing to 1.0 (Ohmic currents)
    std::array<Scalar, 3> branch_lengths{0.01, 0.01, 0.01}; // {t_1, t_2, t_3} internal quartet bridge lengths
    Scalar ohmic_support = 1.0;                // I_Ohm = G_1 / (G_1 + G_2 + G_3)
    Scalar shunt_stress = 0.0;                 // tau_Ohm = G_2 / G_1
};

class Tree {
public:
    std::vector<TreeNode> nodes;
    int32_t root_id = INVALID_INDEX;
    std::vector<int32_t> post_order;
    std::unordered_map<std::string, int32_t> leaf_name_to_id;
    std::vector<WheatstoneBridge> bridges;
    std::vector<int32_t> node_to_bridge_idx;

    bool has_wheatstone_bridges() const {
        return !bridges.empty();
    }

    int32_t bridge_index(int32_t nid) const {
        return (nid >= 0 && static_cast<size_t>(nid) < node_to_bridge_idx.size())
            ? node_to_bridge_idx[nid]
            : -1;
    }

    void rebuild_bridge_index() {
        node_to_bridge_idx.assign(nodes.size(), -1);
        for (size_t k = 0; k < bridges.size(); ++k) {
            int32_t hid = bridges[k].hub_node_id;
            if (hid >= 0 && static_cast<size_t>(hid) < node_to_bridge_idx.size()) {
                node_to_bridge_idx[hid] = static_cast<int32_t>(k);
            }
        }
    }

    void scale_bridge_lengths(Scalar factor) {
        for (auto& wb : bridges) {
            for (int r = 0; r < 3; ++r) {
                wb.branch_lengths[r] *= factor;
            }
        }
    }

    template <typename VecType, typename MatType>
    void eval_internal_node(
        int32_t nid,
        const std::vector<MatType>& P_branches,
        const std::vector<VecType>& node_L,
        const std::vector<double>& node_log_scale,
        VecType& out_L,
        double& out_child_scale_sum,
        const std::vector<std::array<MatType, 3>>* P_bridges = nullptr
    ) const {
        int32_t b_idx = bridge_index(nid);
        if (b_idx >= 0) {
            const auto& wb = bridges[static_cast<size_t>(b_idx)];
            int32_t id_A = wb.children[0];
            int32_t id_B = wb.children[1];
            int32_t id_C = wb.children[2];

            VecType M_A = P_branches[id_A] * node_L[id_A];
            VecType M_B = P_branches[id_B] * node_L[id_B];
            VecType M_C = P_branches[id_C] * node_L[id_C];
            out_child_scale_sum = node_log_scale[id_A] + node_log_scale[id_B] + node_log_scale[id_C];

            auto get_P_br = [&](int r) -> const MatType& {
                if (P_bridges != nullptr) {
                    return (*P_bridges)[static_cast<size_t>(b_idx)][r];
                }
                int32_t bid = wb.bridge_node_id;
                if (bid >= 0 && bid != nid && static_cast<size_t>(bid) < P_branches.size() &&
                    P_branches[bid].rows() == M_A.size()) {
                    return P_branches[bid];
                }
                return P_branches[id_A];
            };

            out_L = VecType::Zero(M_A.size());
            if (wb.weights[0] > 0.0) {
                VecType Y1 = M_A.cwiseProduct(M_B);
                VecType D1 = (get_P_br(0) * Y1).cwiseProduct(M_C);
                out_L.noalias() += wb.weights[0] * D1;
            }
            if (wb.weights[1] > 0.0) {
                VecType Y2 = M_A.cwiseProduct(M_C);
                VecType D2 = (get_P_br(1) * Y2).cwiseProduct(M_B);
                out_L.noalias() += wb.weights[1] * D2;
            }
            if (wb.weights[2] > 0.0) {
                VecType Y3 = M_B.cwiseProduct(M_C);
                VecType D3 = (get_P_br(2) * Y3).cwiseProduct(M_A);
                out_L.noalias() += wb.weights[2] * D3;
            }
        } else {
            out_L.setOnes();
            out_child_scale_sum = 0.0;
            for (int32_t cid : nodes[nid].children) {
                out_L.array() *= (P_branches[cid] * node_L[cid]).array();
                out_child_scale_sum += node_log_scale[cid];
            }
        }
    }

    template <typename VecType, typename MatType, typename ModelType>
    void eval_internal_node_with_model(
        int32_t nid,
        const std::vector<MatType>& P_branches,
        const ModelType& model,
        const std::vector<VecType>& node_L,
        const std::vector<double>& node_log_scale,
        VecType& out_L,
        double& out_child_scale_sum,
        Scalar branch_scale = 1.0
    ) const {
        int32_t b_idx = bridge_index(nid);
        if (b_idx >= 0) {
            const auto& wb = bridges[static_cast<size_t>(b_idx)];
            int32_t id_A = wb.children[0];
            int32_t id_B = wb.children[1];
            int32_t id_C = wb.children[2];

            VecType M_A = P_branches[id_A] * node_L[id_A];
            VecType M_B = P_branches[id_B] * node_L[id_B];
            VecType M_C = P_branches[id_C] * node_L[id_C];
            out_child_scale_sum = node_log_scale[id_A] + node_log_scale[id_B] + node_log_scale[id_C];

            out_L = VecType::Zero(M_A.size());
            if (wb.weights[0] > 0.0) {
                MatType P1 = model.transition_matrix(wb.branch_lengths[0] * branch_scale);
                VecType Y1 = M_A.cwiseProduct(M_B);
                VecType D1 = (P1 * Y1).cwiseProduct(M_C);
                out_L.noalias() += wb.weights[0] * D1;
            }
            if (wb.weights[1] > 0.0) {
                MatType P2 = model.transition_matrix(wb.branch_lengths[1] * branch_scale);
                VecType Y2 = M_A.cwiseProduct(M_C);
                VecType D2 = (P2 * Y2).cwiseProduct(M_B);
                out_L.noalias() += wb.weights[1] * D2;
            }
            if (wb.weights[2] > 0.0) {
                MatType P3 = model.transition_matrix(wb.branch_lengths[2] * branch_scale);
                VecType Y3 = M_B.cwiseProduct(M_C);
                VecType D3 = (P3 * Y3).cwiseProduct(M_A);
                out_L.noalias() += wb.weights[2] * D3;
            }
        } else {
            out_L.setOnes();
            out_child_scale_sum = 0.0;
            for (int32_t cid : nodes[nid].children) {
                out_L.array() *= (P_branches[cid] * node_L[cid]).array();
                out_child_scale_sum += node_log_scale[cid];
            }
        }
    }

    static Tree from_newick_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open Newick file: " + filepath);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return from_newick(buffer.str());
    }

    static Tree from_newick(std::string_view newick_sv) {
        Tree tree;
        std::string newick(newick_sv);

        // Strip trailing semicolon and whitespace
        while (!newick.empty() && (newick.back() == ';' || newick.back() == ' ' || newick.back() == '\n' || newick.back() == '\r')) {
            newick.pop_back();
        }

        // Clean comments [comment], keep model tags {Tag}
        std::string clean;
        bool in_comment = false;
        for (char c : newick) {
            if (c == '[') { in_comment = true; continue; }
            if (c == ']') { in_comment = false; continue; }
            if (!in_comment && c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                clean += c;
            }
        }

        // Parse Newick
        std::stack<int32_t> parent_stack;
        std::string token;
        int32_t current_id = INVALID_INDEX;

        auto create_node = [&]() -> int32_t {
            int32_t id = static_cast<int32_t>(tree.nodes.size());
            TreeNode node;
            node.id = id;
            tree.nodes.push_back(std::move(node));
            return id;
        };

        auto process_token = [&](int32_t target_node, std::string str) {
            if (str.empty()) return;
            std::string tag;
            auto open_brace = str.find('{');
            auto close_brace = str.find('}');
            if (open_brace != std::string::npos && close_brace != std::string::npos && close_brace > open_brace) {
                tag = str.substr(open_brace + 1, close_brace - open_brace - 1);
                str.erase(open_brace, close_brace - open_brace + 1);
            }
            auto colon = str.find(':');
            std::string name;
            Scalar length = 0.0;
            if (colon != std::string::npos) {
                name = str.substr(0, colon);
                try {
                    length = std::stod(str.substr(colon + 1));
                } catch (...) {
                    length = 0.0;
                }
            } else {
                name = str;
            }
            if (name.size() >= 2 && ((name.front() == '\'' && name.back() == '\'') || (name.front() == '"' && name.back() == '"'))) {
                name = name.substr(1, name.size() - 2);
            }
            tree.nodes[target_node].name = name;
            tree.nodes[target_node].branch_length = length;
            tree.nodes[target_node].model_tag = tag;
        };

        for (size_t i = 0; i < clean.size(); ++i) {
            char c = clean[i];
            if (c == '(') {
                int32_t internal_id = create_node();
                if (!parent_stack.empty()) {
                    int32_t p = parent_stack.top();
                    tree.nodes[p].children.push_back(internal_id);
                    tree.nodes[internal_id].parent_id = p;
                }
                parent_stack.push(internal_id);
            } else if (c == ',' || c == ')') {
                if (!token.empty()) {
                    int32_t leaf_id = create_node();
                    tree.nodes[leaf_id].is_leaf = true;
                    process_token(leaf_id, token);
                    token.clear();
                    if (!parent_stack.empty()) {
                        int32_t p = parent_stack.top();
                        tree.nodes[p].children.push_back(leaf_id);
                        tree.nodes[leaf_id].parent_id = p;
                    }
                }
                if (c == ')') {
                    if (!parent_stack.empty()) {
                        current_id = parent_stack.top();
                        parent_stack.pop();
                        // Peek forward for internal node name or branch length
                        size_t j = i + 1;
                        std::string int_token;
                        while (j < clean.size() && clean[j] != ',' && clean[j] != ')' && clean[j] != ';') {
                            int_token += clean[j];
                            ++j;
                        }
                        if (!int_token.empty()) {
                            process_token(current_id, int_token);
                            i = j - 1; // Advance loop
                        }
                    }
                }
            } else {
                token += c;
            }
        }

        if (tree.nodes.empty()) {
            throw std::runtime_error("Failed to parse Newick tree: empty tree");
        }

        // The root is the node with parent_id == INVALID_INDEX
        int node_counter = 1;
        for (auto& node : tree.nodes) {
            if (node.parent_id == INVALID_INDEX) {
                tree.root_id = node.id;
            }
            if (node.name.empty()) {
                node.name = "Node" + std::to_string(node_counter++);
            }
        }

        // Build leaf map and compute post-order traversal
        tree.leaf_name_to_id.clear();
        for (const auto& node : tree.nodes) {
            if (node.is_leaf) {
                tree.leaf_name_to_id[node.name] = node.id;
            }
        }

        tree.compute_post_order();
        return tree;
    }

    size_t num_leaves() const {
        return leaf_name_to_id.size();
    }

    size_t num_nodes() const {
        return nodes.size();
    }

    std::string to_newick(bool include_branch_lengths = true) const {
        if (root_id == INVALID_INDEX || nodes.empty()) return "";
        auto write_node = [&](auto& self, int32_t nid) -> std::string {
            const auto& node = nodes[nid];
            std::string res;
            if (!node.children.empty()) {
                res += "(";
                for (size_t i = 0; i < node.children.size(); ++i) {
                    if (i > 0) res += ",";
                    res += self(self, node.children[i]);
                }
                res += ")";
            }
            res += node.name;
            if (!node.model_tag.empty()) {
                res += "{" + node.model_tag + "}";
            }
            if (include_branch_lengths && nid != root_id) {
                std::ostringstream ss;
                ss << ":" << std::setprecision(8) << node.branch_length;
                res += ss.str();
            }
            return res;
        };
        return write_node(write_node, root_id) + ";";
    }

    void compute_post_order() {
        post_order.clear();
        if (root_id == INVALID_INDEX) return;
        traverse_post_order(root_id);
    }

    void traverse_post_order(int32_t node_id) {
        for (int32_t child_id : nodes[node_id].children) {
            traverse_post_order(child_id);
        }
        post_order.push_back(node_id);
    }
};

} // namespace hyphy::core
