#pragma once

#include "hyphy/core/ohmic_network.hpp"
#include "hyphy/core/likelihood.hpp"

namespace hyphy::core {

class WheatstoneLikelihood {
public:
    struct WheatstoneInsideOutsideResult {
        std::vector<Vector> D;                        // Downward subtree likelihoods [num_nodes]
        std::vector<Vector> V;                        // Upward parent-sibling context [num_nodes]
        std::vector<std::array<Vector, 3>> Y_bridge;  // Cached M_{a_r} .* M_{b_r} [num_bridges][3]
        std::vector<std::array<Vector, 3>> V_bridge;  // Adjoint context for bridge r [num_bridges][3]
        std::vector<std::array<Scalar, 3>> L_bridge;  // Resolution likelihoods L_{p,k,r} [num_bridges][3]
        Scalar likelihood = 0.0;
        Scalar log_likelihood = 0.0;
    };

    // Single-pass GTR log-likelihood on an OhmicNetwork
    static Scalar compute_gtr_log_likelihood(
        const OhmicNetwork& net,
        const Alignment& aln,
        const GTRMatrix& gtr_model,
        Scalar branch_scale = 1.0,
        const std::vector<size_t>* custom_leaf_to_taxon = nullptr
    ) {
        const Tree& tree = net.collapsed_tree;
        size_t num_nodes = tree.num_nodes();
        size_t num_bridges = net.bridges.size();
        size_t num_patterns = aln.patterns.size();

        std::vector<Matrix4> P_branches(num_nodes, Matrix4::Identity());
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                P_branches[node.id] = gtr_model.transition_matrix(node.branch_length * branch_scale);
            }
        }

        std::vector<std::array<Matrix4, 3>> P_bridges(num_bridges);
        for (size_t k = 0; k < num_bridges; ++k) {
            for (int r = 0; r < 3; ++r) {
                if (net.bridges[k].weights[r] > 0.0 || r == 0) {
                    P_bridges[k][r] = gtr_model.transition_matrix(net.bridges[k].branch_lengths[r] * branch_scale);
                } else {
                    P_bridges[k][r] = Matrix4::Identity();
                }
            }
        }

        std::vector<size_t> local_map;
        const std::vector<size_t>& leaf_to_taxon = custom_leaf_to_taxon
            ? *custom_leaf_to_taxon
            : (local_map = LikelihoodEngine::build_leaf_to_taxon_map(tree, aln));

        const double SCALER_THRESH = 1e-150;
        const double SCALER_UP = 1e150;
        const double LOG_SCALER = std::log(SCALER_UP);

        Scalar total_log_l = 0.0;

        #pragma omp parallel for reduction(+:total_log_l) schedule(static)
        for (size_t p = 0; p < num_patterns; ++p) {
            const auto& pattern = aln.patterns[p];
            std::vector<Vector4> node_L(num_nodes, Vector4::Ones());
            std::vector<double> node_log_scale(num_nodes, 0.0);

            for (int32_t nid : tree.post_order) {
                const auto& node = tree.nodes[nid];
                if (node.is_leaf) {
                    size_t t_idx = leaf_to_taxon[nid];
                    if (t_idx != static_cast<size_t>(-1)) {
                        int8_t state = pattern.states[t_idx];
                        if (state >= 0 && state < 4) {
                            node_L[nid].setZero();
                            node_L[nid](state) = 1.0;
                        } else {
                            node_L[nid].setOnes();
                        }
                    } else {
                        node_L[nid].setOnes();
                    }
                    node_log_scale[nid] = 0.0;
                } else {
                    int32_t b_idx = (static_cast<size_t>(nid) < net.node_to_bridge_idx.size())
                        ? net.node_to_bridge_idx[nid] : -1;

                    double child_scale_sum = 0.0;
                    if (b_idx >= 0) {
                        const auto& wb = net.bridges[static_cast<size_t>(b_idx)];
                        int32_t id_A = wb.children[0];
                        int32_t id_B = wb.children[1];
                        int32_t id_C = wb.children[2];

                        Vector4 M_A = P_branches[id_A] * node_L[id_A];
                        Vector4 M_B = P_branches[id_B] * node_L[id_B];
                        Vector4 M_C = P_branches[id_C] * node_L[id_C];
                        child_scale_sum = node_log_scale[id_A] + node_log_scale[id_B] + node_log_scale[id_C];

                        Vector4 D_ohm = Vector4::Zero();
                        if (wb.weights[0] > 0.0) {
                            Vector4 Y1 = M_A.cwiseProduct(M_B);
                            Vector4 D1 = (P_bridges[b_idx][0] * Y1).cwiseProduct(M_C);
                            D_ohm.noalias() += wb.weights[0] * D1;
                        }
                        if (wb.weights[1] > 0.0) {
                            Vector4 Y2 = M_A.cwiseProduct(M_C);
                            Vector4 D2 = (P_bridges[b_idx][1] * Y2).cwiseProduct(M_B);
                            D_ohm.noalias() += wb.weights[1] * D2;
                        }
                        if (wb.weights[2] > 0.0) {
                            Vector4 Y3 = M_B.cwiseProduct(M_C);
                            Vector4 D3 = (P_bridges[b_idx][2] * Y3).cwiseProduct(M_A);
                            D_ohm.noalias() += wb.weights[2] * D3;
                        }
                        node_L[nid] = D_ohm;
                    } else {
                        node_L[nid].setOnes();
                        for (int32_t cid : node.children) {
                            node_L[nid].array() *= (P_branches[cid] * node_L[cid]).array();
                            child_scale_sum += node_log_scale[cid];
                        }
                    }

                    double max_val = node_L[nid].maxCoeff();
                    int boost_count = 0;
                    while (max_val < SCALER_THRESH && max_val > 0.0) {
                        node_L[nid] *= SCALER_UP;
                        max_val *= SCALER_UP;
                        boost_count++;
                    }
                    node_log_scale[nid] = child_scale_sum - boost_count * LOG_SCALER;
                }
            }

            Scalar pattern_likelihood = gtr_model.pi.dot(node_L[tree.root_id]);
            if (pattern_likelihood > 0.0) {
                total_log_l += pattern.weight * (std::log(pattern_likelihood) + node_log_scale[tree.root_id]);
            } else {
                total_log_l += pattern.weight * (-1e20);
            }
        }

        return total_log_l;
    }

    // Single-pass MG94 codon log-likelihood on an OhmicNetwork
    static Scalar compute_mg94_log_likelihood(
        const OhmicNetwork& net,
        const Alignment& aln,
        const MG94Matrix& mg94_model,
        Scalar branch_scale = 1.0,
        const std::vector<size_t>* custom_leaf_to_taxon = nullptr
    ) {
        const Tree& tree = net.collapsed_tree;
        size_t num_nodes = tree.num_nodes();
        size_t num_bridges = net.bridges.size();
        size_t num_patterns = aln.patterns.size();
        int S = static_cast<int>(mg94_model.pi.size());

        std::vector<Matrix> P_branches(num_nodes, Matrix::Identity(S, S));
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id) {
                P_branches[node.id] = mg94_model.transition_matrix(node.branch_length * branch_scale);
            }
        }

        std::vector<std::array<Matrix, 3>> P_bridges(num_bridges);
        for (size_t k = 0; k < num_bridges; ++k) {
            for (int r = 0; r < 3; ++r) {
                if (net.bridges[k].weights[r] > 0.0 || r == 0) {
                    P_bridges[k][r] = mg94_model.transition_matrix(net.bridges[k].branch_lengths[r] * branch_scale);
                } else {
                    P_bridges[k][r] = Matrix::Identity(S, S);
                }
            }
        }

        std::vector<size_t> local_map;
        const std::vector<size_t>& leaf_to_taxon = custom_leaf_to_taxon
            ? *custom_leaf_to_taxon
            : (local_map = LikelihoodEngine::build_leaf_to_taxon_map(tree, aln));

        const double SCALER_THRESH = 1e-150;
        const double SCALER_UP = 1e150;
        const double LOG_SCALER = std::log(SCALER_UP);

        Scalar total_log_l = 0.0;

        #pragma omp parallel
        {
            std::vector<Vector> node_L(num_nodes, Vector::Ones(S));
            std::vector<double> node_log_scale(num_nodes, 0.0);

            #pragma omp for reduction(+:total_log_l) schedule(dynamic)
            for (size_t p = 0; p < num_patterns; ++p) {
                const auto& pattern = aln.patterns[p];

                for (int32_t nid : tree.post_order) {
                    const auto& node = tree.nodes[nid];
                    if (node.is_leaf) {
                        size_t t_idx = leaf_to_taxon[nid];
                        if (t_idx != static_cast<size_t>(-1)) {
                            int8_t state = pattern.states[t_idx];
                            if (state >= 0 && state < S) {
                                node_L[nid].setZero();
                                node_L[nid](state) = 1.0;
                            } else {
                                node_L[nid].setOnes();
                            }
                        } else {
                            node_L[nid].setOnes();
                        }
                        node_log_scale[nid] = 0.0;
                    } else {
                        int32_t b_idx = (static_cast<size_t>(nid) < net.node_to_bridge_idx.size())
                            ? net.node_to_bridge_idx[nid] : -1;

                        double child_scale_sum = 0.0;
                        if (b_idx >= 0) {
                            const auto& wb = net.bridges[static_cast<size_t>(b_idx)];
                            int32_t id_A = wb.children[0];
                            int32_t id_B = wb.children[1];
                            int32_t id_C = wb.children[2];

                            Vector M_A = P_branches[id_A] * node_L[id_A];
                            Vector M_B = P_branches[id_B] * node_L[id_B];
                            Vector M_C = P_branches[id_C] * node_L[id_C];
                            child_scale_sum = node_log_scale[id_A] + node_log_scale[id_B] + node_log_scale[id_C];

                            Vector D_ohm = Vector::Zero(S);
                            if (wb.weights[0] > 0.0) {
                                Vector Y1 = M_A.cwiseProduct(M_B);
                                Vector D1 = (P_bridges[b_idx][0] * Y1).cwiseProduct(M_C);
                                D_ohm.noalias() += wb.weights[0] * D1;
                            }
                            if (wb.weights[1] > 0.0) {
                                Vector Y2 = M_A.cwiseProduct(M_C);
                                Vector D2 = (P_bridges[b_idx][1] * Y2).cwiseProduct(M_B);
                                D_ohm.noalias() += wb.weights[1] * D2;
                            }
                            if (wb.weights[2] > 0.0) {
                                Vector Y3 = M_B.cwiseProduct(M_C);
                                Vector D3 = (P_bridges[b_idx][2] * Y3).cwiseProduct(M_A);
                                D_ohm.noalias() += wb.weights[2] * D3;
                            }
                            node_L[nid] = D_ohm;
                        } else {
                            node_L[nid].setOnes();
                            for (int32_t cid : node.children) {
                                node_L[nid].array() *= (P_branches[cid] * node_L[cid]).array();
                                child_scale_sum += node_log_scale[cid];
                            }
                        }

                        double max_val = node_L[nid].maxCoeff();
                        int boost_count = 0;
                        while (max_val < SCALER_THRESH && max_val > 0.0) {
                            node_L[nid] *= SCALER_UP;
                            max_val *= SCALER_UP;
                            boost_count++;
                        }
                        node_log_scale[nid] = child_scale_sum - boost_count * LOG_SCALER;
                    }
                }

                Scalar pattern_likelihood = mg94_model.pi.dot(node_L[tree.root_id]);
                if (pattern_likelihood > 0.0) {
                    total_log_l += pattern.weight * (std::log(pattern_likelihood) + node_log_scale[tree.root_id]);
                } else {
                    total_log_l += pattern.weight * (-1e20);
                }
            }
        }

        return total_log_l;
    }

    // Exact 2-pass Inside-Outside traversal through backbone + K Wheatstone bridges
    static WheatstoneInsideOutsideResult compute_inside_outside(
        const OhmicNetwork& net,
        const SitePattern& pattern,
        const std::vector<size_t>& leaf_to_taxon,
        const std::vector<Matrix>& P_branches,                 // [num_nodes]
        const std::vector<std::array<Matrix, 3>>& P_bridges,   // [num_bridges][3]
        const Vector& pi,
        int S
    ) {
        const Tree& tree = net.collapsed_tree;
        size_t num_nodes = tree.num_nodes();
        size_t num_bridges = net.bridges.size();

        WheatstoneInsideOutsideResult res;
        res.D.assign(num_nodes, Vector::Zero(S));
        res.V.assign(num_nodes, Vector::Zero(S));
        res.Y_bridge.resize(num_bridges);
        res.V_bridge.resize(num_bridges);
        res.L_bridge.resize(num_bridges, {0.0, 0.0, 0.0});

        std::vector<std::array<Vector, 3>> PY_bridge(num_bridges);
        std::vector<std::array<Vector, 3>> D_bridge_raw(num_bridges);
        for (size_t k = 0; k < num_bridges; ++k) {
            for (int r = 0; r < 3; ++r) {
                res.Y_bridge[k][r] = Vector::Zero(S);
                res.V_bridge[k][r] = Vector::Zero(S);
                PY_bridge[k][r] = Vector::Zero(S);
                D_bridge_raw[k][r] = Vector::Zero(S);
            }
        }

        std::vector<Vector> P_D(num_nodes, Vector::Zero(S));
        std::vector<double> B(num_nodes, 1.0);
        std::vector<double> node_log_scale(num_nodes, 0.0);

        const double SCALER_THRESH = 1e-150;
        const double SCALER_UP = 1e150;
        const double LOG_SCALER = std::log(SCALER_UP);

        // 1. Post-order (bottom-up) Inside pass
        for (int32_t nid : tree.post_order) {
            const auto& node = tree.nodes[nid];
            if (node.is_leaf) {
                size_t t_idx = leaf_to_taxon[nid];
                if (t_idx != static_cast<size_t>(-1)) {
                    int8_t state = pattern.states[t_idx];
                    if (state >= 0 && state < S) {
                        res.D[nid].setZero();
                        res.D[nid](state) = 1.0;
                    } else {
                        res.D[nid].setOnes();
                    }
                } else {
                    res.D[nid].setOnes();
                }
                node_log_scale[nid] = 0.0;
                B[nid] = 1.0;
            } else {
                int32_t b_idx = (static_cast<size_t>(nid) < net.node_to_bridge_idx.size())
                    ? net.node_to_bridge_idx[nid] : -1;

                double child_scale_sum = 0.0;
                if (b_idx >= 0) {
                    size_t k = static_cast<size_t>(b_idx);
                    const auto& wb = net.bridges[k];
                    int32_t id_A = wb.children[0];
                    int32_t id_B = wb.children[1];
                    int32_t id_C = wb.children[2];

                    P_D[id_A] = P_branches[id_A] * res.D[id_A];
                    P_D[id_B] = P_branches[id_B] * res.D[id_B];
                    P_D[id_C] = P_branches[id_C] * res.D[id_C];
                    child_scale_sum = node_log_scale[id_A] + node_log_scale[id_B] + node_log_scale[id_C];

                    // Resolution 1: (A, B) | C
                    res.Y_bridge[k][0] = P_D[id_A].cwiseProduct(P_D[id_B]);
                    PY_bridge[k][0] = P_bridges[k][0] * res.Y_bridge[k][0];
                    D_bridge_raw[k][0] = PY_bridge[k][0].cwiseProduct(P_D[id_C]);

                    // Resolution 2: (A, C) | B
                    res.Y_bridge[k][1] = P_D[id_A].cwiseProduct(P_D[id_C]);
                    PY_bridge[k][1] = P_bridges[k][1] * res.Y_bridge[k][1];
                    D_bridge_raw[k][1] = PY_bridge[k][1].cwiseProduct(P_D[id_B]);

                    // Resolution 3: (B, C) | A
                    res.Y_bridge[k][2] = P_D[id_B].cwiseProduct(P_D[id_C]);
                    PY_bridge[k][2] = P_bridges[k][2] * res.Y_bridge[k][2];
                    D_bridge_raw[k][2] = PY_bridge[k][2].cwiseProduct(P_D[id_A]);

                    res.D[nid] = wb.weights[0] * D_bridge_raw[k][0]
                               + wb.weights[1] * D_bridge_raw[k][1]
                               + wb.weights[2] * D_bridge_raw[k][2];
                } else {
                    res.D[nid].setOnes();
                    for (int32_t cid : node.children) {
                        P_D[cid] = P_branches[cid] * res.D[cid];
                        res.D[nid].array() *= P_D[cid].array();
                        child_scale_sum += node_log_scale[cid];
                    }
                }

                double max_val = res.D[nid].maxCoeff();
                int boost_count = 0;
                double boost_mult = 1.0;
                while (max_val < SCALER_THRESH && max_val > 0.0) {
                    res.D[nid] *= SCALER_UP;
                    max_val *= SCALER_UP;
                    boost_mult *= SCALER_UP;
                    boost_count++;
                }
                B[nid] = boost_mult;
                node_log_scale[nid] = child_scale_sum - boost_count * LOG_SCALER;
            }
        }

        res.likelihood = res.D[tree.root_id].dot(pi);
        if (res.likelihood > 0.0) {
            res.log_likelihood = std::log(res.likelihood) + node_log_scale[tree.root_id];
        } else {
            res.log_likelihood = -1e20;
        }

        // 2. Pre-order (top-down) Outside pass
        std::vector<Vector> U(num_nodes, Vector::Zero(S));
        U[tree.root_id] = pi;

        for (auto it = tree.post_order.rbegin(); it != tree.post_order.rend(); ++it) {
            int32_t uid = *it;
            const auto& node = tree.nodes[uid];
            if (node.is_leaf) continue;

            int32_t b_idx = (static_cast<size_t>(uid) < net.node_to_bridge_idx.size())
                ? net.node_to_bridge_idx[uid] : -1;

            if (b_idx >= 0) {
                size_t k = static_cast<size_t>(b_idx);
                const auto& wb = net.bridges[k];
                int32_t id_A = wb.children[0];
                int32_t id_B = wb.children[1];
                int32_t id_C = wb.children[2];

                Vector U_tilde = B[uid] * U[uid];

                // Resolution-specific pattern likelihoods L_{p,k,r} = \tilde{U}_u^T D_u^{(r), raw}
                for (int r = 0; r < 3; ++r) {
                    res.L_bridge[k][r] = U_tilde.dot(D_bridge_raw[k][r]);
                }

                // Pre-bridge upward context vectors V_{k,r}^{bridge} = w_{k,r} (\tilde{U}_u .* M_{c_r})
                Vector U_MC = U_tilde.cwiseProduct(P_D[id_C]);
                Vector U_MB = U_tilde.cwiseProduct(P_D[id_B]);
                Vector U_MA = U_tilde.cwiseProduct(P_D[id_A]);

                res.V_bridge[k][0] = wb.weights[0] * U_MC;
                res.V_bridge[k][1] = wb.weights[1] * U_MB;
                res.V_bridge[k][2] = wb.weights[2] * U_MA;

                // Pull back across bridge transition matrices P_{k,r}^T
                Vector Z1 = P_bridges[k][0].transpose() * U_MC;
                Vector Z2 = P_bridges[k][1].transpose() * U_MB;
                Vector Z3 = P_bridges[k][2].transpose() * U_MA;

                // Exact superposed top-down context vectors V_A, V_B, V_C (Section 3.5.3)
                res.V[id_A] = wb.weights[0] * Z1.cwiseProduct(P_D[id_B])
                            + wb.weights[1] * Z2.cwiseProduct(P_D[id_C])
                            + wb.weights[2] * U_tilde.cwiseProduct(PY_bridge[k][2]);

                res.V[id_B] = wb.weights[0] * Z1.cwiseProduct(P_D[id_A])
                            + wb.weights[1] * U_tilde.cwiseProduct(PY_bridge[k][1])
                            + wb.weights[2] * Z3.cwiseProduct(P_D[id_C]);

                res.V[id_C] = wb.weights[0] * U_tilde.cwiseProduct(PY_bridge[k][0])
                            + wb.weights[1] * Z2.cwiseProduct(P_D[id_A])
                            + wb.weights[2] * Z3.cwiseProduct(P_D[id_B]);

                U[id_A] = P_branches[id_A].transpose() * res.V[id_A];
                U[id_B] = P_branches[id_B].transpose() * res.V[id_B];
                U[id_C] = P_branches[id_C].transpose() * res.V[id_C];
            } else {
                size_t num_children = node.children.size();
                for (size_t i = 0; i < num_children; ++i) {
                    int32_t vid = node.children[i];
                    Vector v_msg = U[uid];
                    for (size_t j = 0; j < num_children; ++j) {
                        if (i != j) {
                            v_msg.array() *= P_D[node.children[j]].array();
                        }
                    }
                    v_msg *= B[uid];
                    res.V[vid] = v_msg;
                    U[vid] = P_branches[vid].transpose() * v_msg;
                }
            }
        }

        return res;
    }

    struct JointWheatstoneMG94Gradients {
        Scalar log_likelihood = 0.0;
        std::vector<Scalar> grad_branches;                      // [num_nodes]
        std::vector<std::array<Scalar, 3>> grad_bridge_lengths; // [num_bridges][3]
        std::vector<std::array<Scalar, 3>> grad_bridge_weights; // [num_bridges][3]
        Scalar grad_omega = 0.0;
        Scalar grad_theta_AC = 0.0;
        Scalar grad_theta_AT = 0.0;
        Scalar grad_theta_CG = 0.0;
        Scalar grad_theta_CT = 0.0;
        Scalar grad_theta_GT = 0.0;
    };

    // Exact joint analytical gradients for all backbone branches, bridge branches,
    // bridge weights, and global MG94 parameters via Inside-Outside + Fréchet derivatives
    static JointWheatstoneMG94Gradients compute_joint_frechet_gradients(
        const OhmicNetwork& net,
        const Alignment& aln,
        const MG94Parameters& params,
        const std::vector<size_t>* custom_leaf_to_taxon = nullptr
    ) {
        const Tree& tree = net.collapsed_tree;
        const GeneticCode& gcode = *(aln.code ? aln.code : GeneticCode::universal());
        int S = gcode.num_sense_codons;
        size_t num_nodes = tree.num_nodes();
        size_t num_bridges = net.bridges.size();
        size_t num_patterns = aln.patterns.size();

        MG94Matrix mg;
        mg.update(params, aln.pos_nuc_frequencies, aln.codon_frequencies_f3x4, gcode);
        Scalar mu = std::max(mg.scale_factor, 1e-12);
        Scalar conv = 3.0 / mu;

        std::vector<Scalar> tau_branches(num_nodes, 0.0);
        std::vector<Matrix> P_branches(num_nodes, Matrix::Identity(S, S));
        std::vector<Matrix> dP_branches(num_nodes, Matrix::Zero(S, S));

        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id && (node.is_leaf || !node.children.empty())) {
                int32_t bid = node.id;
                tau_branches[bid] = node.branch_length * conv;
                P_branches[bid] = mg.transition_matrix(tau_branches[bid]);
                dP_branches[bid] = mg.Q * P_branches[bid];
            }
        }

        std::vector<std::array<Scalar, 3>> tau_bridges(num_bridges, {0.0, 0.0, 0.0});
        std::vector<std::array<Matrix, 3>> P_bridges(num_bridges);
        std::vector<std::array<Matrix, 3>> dP_bridges(num_bridges);

        for (size_t k = 0; k < num_bridges; ++k) {
            for (int r = 0; r < 3; ++r) {
                tau_bridges[k][r] = net.bridges[k].branch_lengths[r] * conv;
                P_bridges[k][r] = mg.transition_matrix(tau_bridges[k][r]);
                dP_bridges[k][r] = mg.Q * P_bridges[k][r];
            }
        }

        std::vector<size_t> local_map;
        const std::vector<size_t>& leaf_to_taxon = custom_leaf_to_taxon
            ? *custom_leaf_to_taxon
            : (local_map = LikelihoodEngine::build_leaf_to_taxon_map(tree, aln));

        Scalar total_log_l = 0.0;
        std::vector<Scalar> dL_dtau_branches(num_nodes, 0.0);
        std::vector<Matrix> A_branches(num_nodes, Matrix::Zero(S, S));

        std::vector<std::array<Scalar, 3>> dL_dtau_bridges(num_bridges, {0.0, 0.0, 0.0});
        std::vector<std::array<Scalar, 3>> dL_dw_bridges(num_bridges, {0.0, 0.0, 0.0});
        std::vector<std::array<Matrix, 3>> A_bridges(num_bridges);
        for (size_t k = 0; k < num_bridges; ++k) {
            for (int r = 0; r < 3; ++r) {
                A_bridges[k][r] = Matrix::Zero(S, S);
            }
        }

        #pragma omp parallel
        {
            Scalar local_ll = 0.0;
            std::vector<Scalar> local_dL_b(num_nodes, 0.0);
            std::vector<Matrix> local_A_b(num_nodes, Matrix::Zero(S, S));

            std::vector<std::array<Scalar, 3>> local_dL_br(num_bridges, {0.0, 0.0, 0.0});
            std::vector<std::array<Scalar, 3>> local_dL_w(num_bridges, {0.0, 0.0, 0.0});
            std::vector<std::array<Matrix, 3>> local_A_br(num_bridges);
            for (size_t k = 0; k < num_bridges; ++k) {
                for (int r = 0; r < 3; ++r) {
                    local_A_br[k][r] = Matrix::Zero(S, S);
                }
            }

            #pragma omp for schedule(dynamic)
            for (size_t p = 0; p < num_patterns; ++p) {
                const auto& pattern = aln.patterns[p];
                auto io = compute_inside_outside(
                    net, pattern, leaf_to_taxon, P_branches, P_bridges, aln.codon_frequencies_f3x4, S
                );
                Scalar L = io.likelihood;
                if (L > 0.0) {
                    local_ll += pattern.weight * io.log_likelihood;
                    Scalar inv_L = pattern.weight / L;

                    // Backbone branch accumulators
                    for (const auto& node : tree.nodes) {
                        if (node.id != tree.root_id && (node.is_leaf || !node.children.empty())) {
                            int32_t bid = node.id;
                            Scalar dL_dt = io.V[bid].dot(dP_branches[bid] * io.D[bid]);
                            local_dL_b[bid] += inv_L * dL_dt;
                            local_A_b[bid].noalias() += inv_L * (io.V[bid] * io.D[bid].transpose());
                        }
                    }

                    // Bridge branch & weight accumulators
                    for (size_t k = 0; k < num_bridges; ++k) {
                        for (int r = 0; r < 3; ++r) {
                            local_dL_w[k][r] += inv_L * io.L_bridge[k][r];
                            if (net.bridges[k].weights[r] > 0.0 || r == 0) {
                                Scalar dL_dt = io.V_bridge[k][r].dot(dP_bridges[k][r] * io.Y_bridge[k][r]);
                                local_dL_br[k][r] += inv_L * dL_dt;
                                local_A_br[k][r].noalias() += inv_L * (io.V_bridge[k][r] * io.Y_bridge[k][r].transpose());
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                total_log_l += local_ll;
                for (size_t i = 0; i < num_nodes; ++i) {
                    dL_dtau_branches[i] += local_dL_b[i];
                    A_branches[i] += local_A_b[i];
                }
                for (size_t k = 0; k < num_bridges; ++k) {
                    for (int r = 0; r < 3; ++r) {
                        dL_dtau_bridges[k][r] += local_dL_br[k][r];
                        dL_dw_bridges[k][r] += local_dL_w[k][r];
                        A_bridges[k][r] += local_A_br[k][r];
                    }
                }
            }
        }

        // Spectral Loewner matrix accumulation across BOTH backbone branches and bridge branches
        Matrix H = Matrix::Zero(S, S);
        Scalar sum_tau_dL_dtau = 0.0;

        auto accumulate_loewner = [&](Scalar tb, Scalar dL_dt, const Matrix& A_mat) {
            sum_tau_dL_dtau += tb * dL_dt;
            Matrix A_tilde = mg.U_pi.transpose() * A_mat * mg.V_pi.transpose();

            Matrix E(S, S);
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < S; ++j) {
                    Scalar diff = mg.eigenvalues(i) - mg.eigenvalues(j);
                    if (std::abs(diff) < 1e-9) {
                        E(i, j) = tb * std::exp(mg.eigenvalues(i) * tb);
                    } else {
                        E(i, j) = (std::exp(mg.eigenvalues(i) * tb) - std::exp(mg.eigenvalues(j) * tb)) / diff;
                    }
                }
            }
            H.noalias() += A_tilde.cwiseProduct(E);
        };

        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id && (node.is_leaf || !node.children.empty())) {
                int32_t bid = node.id;
                accumulate_loewner(tau_branches[bid], dL_dtau_branches[bid], A_branches[bid]);
            }
        }

        for (size_t k = 0; k < num_bridges; ++k) {
            for (int r = 0; r < 3; ++r) {
                if (net.bridges[k].weights[r] > 0.0) {
                    accumulate_loewner(tau_bridges[k][r], dL_dtau_bridges[k][r], A_bridges[k][r]);
                }
            }
        }

        auto compute_global_grad = [&](const Matrix& dQ_k) -> Scalar {
            Matrix M_k = mg.V_pi * dQ_k * mg.U_pi;
            Scalar part1 = (M_k.cwiseProduct(H)).sum();
            Scalar dmu_k = 0.0;
            for (int i = 0; i < S; ++i) {
                dmu_k -= mg.pi(i) * dQ_k(i, i);
            }
            Scalar part2 = -(dmu_k / mu) * sum_tau_dL_dtau;
            return part1 + part2;
        };

        Scalar r_nuc[4][4] = {
            {0.0, params.theta_AC, 1.0, params.theta_AT},
            {params.theta_AC, 0.0, params.theta_CG, params.theta_CT},
            {1.0, params.theta_CG, 0.0, params.theta_GT},
            {params.theta_AT, params.theta_CT, params.theta_GT, 0.0}
        };

        Matrix dQ_omega = Matrix::Zero(S, S);
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                if (i == j) continue;
                const auto& diff = gcode.diff_matrix[i][j];
                if (diff.num_diffs == 1 && !diff.synonymous) {
                    Scalar nuc_rate = r_nuc[diff.diff_from[0]][diff.diff_to[0]];
                    Scalar target_nuc_freq = aln.pos_nuc_frequencies(diff.diff_pos[0], diff.diff_to[0]);
                    dQ_omega(i, j) = 1.0 * nuc_rate * target_nuc_freq;
                }
            }
            dQ_omega(i, i) = -dQ_omega.row(i).sum();
        }

        auto build_dQ_nuc = [&](int u, int v) -> Matrix {
            Matrix dQ = Matrix::Zero(S, S);
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < S; ++j) {
                    if (i == j) continue;
                    const auto& diff = gcode.diff_matrix[i][j];
                    if (diff.num_diffs == 1) {
                        int from = diff.diff_from[0];
                        int to = diff.diff_to[0];
                        if ((from == u && to == v) || (from == v && to == u)) {
                            Scalar rate_mod = diff.synonymous ? params.alpha : params.beta;
                            Scalar target_nuc_freq = aln.pos_nuc_frequencies(diff.diff_pos[0], to);
                            dQ(i, j) = rate_mod * 1.0 * target_nuc_freq;
                        }
                    }
                }
                dQ(i, i) = -dQ.row(i).sum();
            }
            return dQ;
        };

        JointWheatstoneMG94Gradients res;
        res.log_likelihood = total_log_l;
        res.grad_branches.assign(num_nodes, 0.0);
        for (const auto& node : tree.nodes) {
            if (node.id != tree.root_id && (node.is_leaf || !node.children.empty())) {
                res.grad_branches[node.id] = conv * dL_dtau_branches[node.id];
            }
        }

        res.grad_bridge_lengths.resize(num_bridges, {0.0, 0.0, 0.0});
        res.grad_bridge_weights = dL_dw_bridges;
        for (size_t k = 0; k < num_bridges; ++k) {
            for (int r = 0; r < 3; ++r) {
                res.grad_bridge_lengths[k][r] = conv * dL_dtau_bridges[k][r];
            }
            // Also populate grad_branches[bridge_node_id] with primary bridge gradient
            // when bridge_node_id was collapsed or converted into a zero-branch-length root child hub
            int32_t bid = net.bridges[k].bridge_node_id;
            if (bid >= 0 && static_cast<size_t>(bid) < num_nodes) {
                if (tree.nodes[bid].children.empty() || net.bridges[k].hub_node_id == bid) {
                    res.grad_branches[bid] = res.grad_bridge_lengths[k][0];
                }
            }
        }

        res.grad_omega = compute_global_grad(dQ_omega);
        res.grad_theta_AC = compute_global_grad(build_dQ_nuc(0, 1));
        res.grad_theta_AT = compute_global_grad(build_dQ_nuc(0, 3));
        res.grad_theta_CG = compute_global_grad(build_dQ_nuc(1, 2));
        res.grad_theta_CT = compute_global_grad(build_dQ_nuc(1, 3));
        res.grad_theta_GT = compute_global_grad(build_dQ_nuc(2, 3));

        return res;
    }
};

} // namespace hyphy::core
