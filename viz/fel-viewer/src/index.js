// Main Entry Point: HyPhyFELViewer Class & <hyphy-fel-viewer> Web Component

import { FELData } from './data/fel-data.js';
import { TreeCanvas } from './components/tree-canvas.js';
import { AlignmentCanvas } from './components/alignment-canvas.js';
import { TrackHeader } from './components/track-header.js';
import { BirdsEyeOverview } from './components/birds-eye-overview.js';
import { Inspector } from './components/inspector.js';
import { Toolbar } from './components/toolbar.js';
import { NativeBridge } from './bridge/native-bridge.js';
import './styles/viewer.css';

export class HyPhyFELViewer {
  constructor(container, options = {}) {
    this.container = container;
    this.options = Object.assign({
      theme: 'light',
      rowHeight: 22,
      displayMode: 'amino_acid',
      colorScheme: 'difference',
      pThreshold: 0.1,
      onSiteSelect: null,
      onBranchSelect: null
    }, options);

    this.data = null;
    this.selectedSite = 1;
    this.showAncestors = false;
    this.currentTaxaOrder = [];
    this.bridge = new NativeBridge(this.container);

    this.initDOM();
    this.initSplitter();
    this.initResizeObserver();
  }

  initDOM() {
    this.container.classList.add('hyphy-fel-container');
    if (this.options.theme === 'light') {
      this.container.classList.add('hyphy-theme-light');
    } else {
      this.container.classList.add('hyphy-theme-dark');
    }

    // 1. Toolbar
    this.toolbar = new Toolbar(this.container, {
      onModeChange: (mode) => this.setDisplayMode(mode),
      onSchemeChange: (scheme) => this.setColorScheme(scheme),
      onSiteJump: (site) => this.selectSite(site),
      onThresholdChange: (p) => this.setPThreshold(p),
      onFileLoad: (file) => this.loadFile(file),
      onSampleSelect: (sampleName) => this.loadSample(sampleName),
      onThemeToggle: () => this.toggleTheme(),
      onAncestorsToggle: (show) => this.setAncestorsVisible(show),
      onOverviewToggle: () => this.toggleOverview()
    });

    // 2. Bird's Eye Overview: Whole-Gene Selection Landscape & Minimap
    this.overview = new BirdsEyeOverview(this.container, {
      pThreshold: this.options.pThreshold,
      colorScheme: this.options.colorScheme,
      displayMode: this.options.displayMode,
      onModeChange: () => this.resize(),
      onSiteSelect: (site, siteResult) => this.selectSite(site),
      onViewportChange: (startCol, startRow) => {
        const colW = this.alignViewer.getColWidth();
        const rowH = this.options.rowHeight;
        const newSl = startCol * colW;
        const newSt = startRow * rowH;
        this.alignViewer.setScroll(newSl, newSt);
        this.treeViewer.setScrollTop(this.alignViewer.scrollTop);
        this.trackHeader.setScrollLeft(this.alignViewer.scrollLeft);
      }
    });

    // 3. Main Stage (Splitter View)
    this.stage = document.createElement('div');
    this.stage.className = 'hyphy-main-stage';
    this.container.appendChild(this.stage);

    // Left Pane: Tree
    this.treePane = document.createElement('div');
    this.treePane.className = 'hyphy-tree-pane';
    this.stage.appendChild(this.treePane);

    this.treeHeader = document.createElement('div');
    this.treeHeader.className = 'hyphy-tree-header';
    this.treeHeader.innerHTML = `
      <div class="tree-header-title">Phylogeny</div>
      <div class="tree-header-sub" id="tree-sub">Loading...</div>
    `;
    this.treePane.appendChild(this.treeHeader);

    this.treeContainer = document.createElement('div');
    this.treeContainer.className = 'tree-canvas-container';
    this.treePane.appendChild(this.treeContainer);

    this.treeViewer = new TreeCanvas(this.treeContainer, {
      rowHeight: this.options.rowHeight,
      onBranchSelect: (branch) => this.selectBranch(branch),
      onLeafHover: (leafName, leafIdx) => {
        // Sync row hover highlight
      }
    });

    // Draggable Splitter
    this.splitter = document.createElement('div');
    this.splitter.className = 'hyphy-splitter';
    this.stage.appendChild(this.splitter);

    // Right Pane: Tracks & Alignment
    this.alignPane = document.createElement('div');
    this.alignPane.className = 'hyphy-alignment-pane';
    this.stage.appendChild(this.alignPane);

    // Sticky Tracks Header
    this.tracksContainer = document.createElement('div');
    this.tracksContainer.className = 'hyphy-tracks-container';
    this.alignPane.appendChild(this.tracksContainer);

    this.trackHeader = new TrackHeader(this.tracksContainer, {
      pThreshold: this.options.pThreshold,
      displayMode: this.options.displayMode,
      onSiteSelect: (site, siteResult) => this.selectSite(site)
    });

    // Alignment Canvas
    this.alignContainer = document.createElement('div');
    this.alignContainer.className = 'hyphy-alignment-container';
    this.alignPane.appendChild(this.alignContainer);

    this.alignViewer = new AlignmentCanvas(this.alignContainer, {
      rowHeight: this.options.rowHeight,
      displayMode: this.options.displayMode,
      colorScheme: this.options.colorScheme,
      onSiteSelect: (site, siteResult) => this.selectSite(site),
      onScroll: (scrollLeft, scrollTop, range) => {
        // Synchronize Tree vertically & Tracks Header horizontally
        this.treeViewer.setScrollTop(scrollTop);
        this.trackHeader.setScrollLeft(scrollLeft);
        if (range && this.data) {
          this.overview.setViewport(
            range.startCol,
            range.endCol,
            range.startRow,
            range.endRow,
            this.data.numSites,
            this.currentTaxaOrder.length
          );
        }
      }
    });

    // Wheel listener on Tree to scroll Alignment vertically in lockstep
    this.treeContainer.addEventListener('wheel', (e) => {
      e.preventDefault();
      const newSt = this.alignViewer.scrollTop + e.deltaY;
      this.alignViewer.setScroll(this.alignViewer.scrollLeft, newSt);
      this.treeViewer.setScrollTop(this.alignViewer.scrollTop);
    }, { passive: false });

    // 3. Detail Inspector Panel
    this.inspector = new Inspector(this.container, {
      onBranchHighlight: (branch) => {
        this.treeViewer.highlightBranch(branch);
      }
    });
  }

  load(jsonData) {
    this.data = new FELData(jsonData);

    // Update Tree Subtitle
    const treeSub = this.treeHeader.querySelector('#tree-sub');
    if (treeSub) {
      const taxaCount = this.data.taxa.length;
      const sitesCount = this.data.numSites;
      const methodStr = this.data.isMEME ? 'MEME (Episodic)' : 'FEL (Pervasive)';
      const testedStr = this.data.hasBackground 
        ? `${this.data.numTested} test, ${this.data.numBackground} bg`
        : `All branches`;
      treeSub.textContent = `${methodStr} • ${taxaCount} taxa • ${sitesCount} sites • ${testedStr}`;
    }

    if (this.data.isMEME) {
      this.toolbar.setDatasetName('cd2_meme');
    }

    // 1. Pass to Tree Viewer
    this.treeViewer.setNewick(this.data.treeNewick, this.data.treeNodes);

    // 2. Reconstruct Inferred Ancestral Sequences from Tree & Events
    this.data.reconstructAncestralSequences(this.treeViewer.root);

    // 3. Compute Taxa Display Order (Extant or Extant + Ancestors)
    this.updateTaxaOrder();

    // 4. Pass Data to Alignment, Tracks & Overview
    this.alignViewer.setData(this.data, this.currentTaxaOrder);
    this.trackHeader.setData(this.data);
    this.overview.setData(this.data, this.currentTaxaOrder);

    // 5. Default Select First Site or First Significant Site
    let initialSite = 1;
    const firstSig = this.data.siteResults.find(r => r.pValue <= this.options.pThreshold);
    if (firstSig) initialSite = firstSig.site;

    this.selectSite(initialSite);
    this.resize();
  }

  updateTaxaOrder() {
    if (!this.data) return;
    const leaves = this.treeViewer.leafOrder.length > 0 
      ? this.treeViewer.leafOrder 
      : this.data.taxa;

    if (this.showAncestors && this.data.ancestralTaxa?.length > 0) {
      this.currentTaxaOrder = [ ...leaves, ...this.data.ancestralTaxa ];
    } else {
      this.currentTaxaOrder = leaves;
    }
    this.overview?.setTaxaOrder(this.currentTaxaOrder);
  }

  setAncestorsVisible(show) {
    this.showAncestors = show;
    this.updateTaxaOrder();
    this.treeViewer.setShowAncestors(show, this.data?.ancestralTaxa || []);
    this.alignViewer.setTaxaOrder(this.currentTaxaOrder);
    this.overview?.setTaxaOrder(this.currentTaxaOrder);
    this.resize();
  }

  selectSite(siteIndex) {
    if (!this.data) return;
    const clampedSite = Math.max(1, Math.min(siteIndex, this.data.numSites));
    this.selectedSite = clampedSite;

    const siteResult = this.data.siteResults[clampedSite - 1];
    const consensusAA = this.data.consensusAA[clampedSite - 1] || '-';
    const siteEvents = this.data.eventsBySite[clampedSite] || [];

    // 1. Update Alignment column cursor
    this.alignViewer.setSelectedSite(clampedSite);

    // 2. Update Track Header cursor
    this.trackHeader.setSelectedSite(clampedSite);

    // 3. Update Overview cursor needle
    this.overview.setSelectedSite(clampedSite);

    // 4. Highlight mutations on Phylogenetic Tree
    this.treeViewer.setActiveSiteEvents(siteEvents);
    this.treeViewer.setActiveSiteEBF(siteResult?.branchEBF || {});

    // 5. Update Inspector panel
    this.inspector.setSite(siteResult, consensusAA, this.data.isMEME);

    // 6. Scroll Alignment into view if needed
    const colW = this.alignViewer.getColWidth();
    const siteX = (clampedSite - 1) * colW;
    const curLeft = this.alignViewer.scrollLeft;
    const viewW = this.alignViewer.width || 600;

    if (siteX < curLeft) {
      this.alignViewer.setScroll(siteX - 40, this.alignViewer.scrollTop);
      this.trackHeader.setScrollLeft(this.alignViewer.scrollLeft);
    } else if (siteX + colW > curLeft + viewW) {
      this.alignViewer.setScroll(siteX - viewW + colW + 40, this.alignViewer.scrollTop);
      this.trackHeader.setScrollLeft(this.alignViewer.scrollLeft);
    }

    // 7. Notify Bridge (macOS Cocoa / Host DOM)
    this.bridge.notifySiteSelected(siteResult, siteEvents);
    if (this.options.onSiteSelect) {
      this.options.onSiteSelect(siteResult);
    }
  }

  selectBranch(branchName) {
    const siteEvents = (this.data?.eventsByBranch[branchName] || []).filter(
      ev => ev.site === this.selectedSite
    );
    this.bridge.notifyBranchSelected(branchName, siteEvents);
    if (this.options.onBranchSelect) {
      this.options.onBranchSelect(branchName);
    }
  }

  setDisplayMode(mode) {
    this.options.displayMode = mode;
    this.alignViewer.setDisplayMode(mode);
    this.trackHeader.setDisplayMode(mode);
    this.overview.setDisplayMode(mode);
  }

  setColorScheme(scheme) {
    this.options.colorScheme = scheme;
    this.alignViewer.setColorScheme(scheme);
    this.overview.setColorScheme(scheme);
  }

  setPThreshold(th) {
    this.options.pThreshold = th;
    this.trackHeader.setPThreshold(th);
    this.overview.setPThreshold(th);
  }

  toggleOverview() {
    if (!this.overview) return;
    const panel = this.overview.wrapper;
    if (panel.style.display === 'none') {
      panel.style.display = 'flex';
    } else {
      panel.style.display = 'none';
    }
    this.resize();
  }

  toggleTheme() {
    if (this.container.classList.contains('hyphy-theme-light')) {
      this.setTheme('dark');
    } else {
      this.setTheme('light');
    }
  }

  setTheme(theme) {
    this.options.theme = theme;
    this.container.classList.remove('hyphy-theme-light', 'hyphy-theme-dark');
    this.container.classList.add(`hyphy-theme-${theme}`);
    this.bridge.notifyThemeChanged(theme);
    this.resize();
  }

  loadFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      if (file.name.endsWith('.json') || text.trim().startsWith('{')) {
        try {
          const json = JSON.parse(text);
          this.load(json);
        } catch (err) {
          alert('Failed to parse JSON file: ' + err.message);
        }
      } else {
        // External FASTA / NEXUS alignment file
        if (this.data) {
          const ok = this.data.loadExternalAlignment(text);
          if (ok) {
            this.alignViewer.setData(this.data);
            alert(`Loaded alignment for ${this.data.numTaxa} taxa.`);
          } else {
            alert('Failed to parse alignment format (FASTA/NEXUS).');
          }
        } else {
          alert('Please load a HyPhy FEL JSON file first before adding external alignment.');
        }
      }
    };
    reader.readAsText(file);
  }

  loadSample(name) {
    let url = '/samples/bglobin.json';
    if (name === 'fig4e') url = '/samples/fig4e.json';
    else if (name === 'cd2_meme') url = '/samples/cd2_meme.json';
    fetch(url)
      .then(res => res.json())
      .then(data => this.load(data))
      .catch(err => console.error('Failed to load sample:', err));
  }

  initSplitter() {
    let isDragging = false;
    let startX = 0;
    let startWidth = 0;

    this.splitter.addEventListener('mousedown', (e) => {
      isDragging = true;
      startX = e.clientX;
      startWidth = this.treePane.getBoundingClientRect().width;
      this.splitter.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const newWidth = Math.max(160, Math.min(startWidth + dx, 600));
      this.treePane.style.flex = `0 0 ${newWidth}px`;
      this.resize();
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        this.splitter.classList.remove('dragging');
        document.body.style.cursor = '';
      }
    });
  }

  initResizeObserver() {
    if (window.ResizeObserver) {
      this.ro = new ResizeObserver(() => {
        this.resize();
      });
      this.ro.observe(this.container);
    } else {
      window.addEventListener('resize', () => this.resize());
    }
  }

  resize() {
    const ovRect = this.container.getBoundingClientRect();
    if (ovRect.width > 0 && this.overview) {
      this.overview.resize(ovRect.width);
    }

    const treeRect = this.treeContainer.getBoundingClientRect();
    if (treeRect.width > 0 && treeRect.height > 0) {
      this.treeViewer.resize(treeRect.width, treeRect.height);
    }

    const tracksRect = this.tracksContainer.getBoundingClientRect();
    if (tracksRect.width > 0) {
      this.trackHeader.resize(tracksRect.width);
    }

    const alignRect = this.alignContainer.getBoundingClientRect();
    if (alignRect.width > 0 && alignRect.height > 0) {
      this.alignViewer.resize(alignRect.width, alignRect.height);
      const range = this.alignViewer.getVisibleRange();
      if (range && this.data && this.overview) {
        this.overview.setViewport(
          range.startCol,
          range.endCol,
          range.startRow,
          range.endRow,
          this.data.numSites,
          this.currentTaxaOrder.length
        );
      }
    }
  }
}

// Register as Standard W3C Custom Element (<hyphy-fel-viewer>)
if (typeof customElements !== 'undefined' && !customElements.get('hyphy-fel-viewer')) {
  class HyPhyFELViewerElement extends HTMLElement {
    connectedCallback() {
      if (!this.viewer) {
        const theme = this.getAttribute('theme') || 'dark';
        this.viewer = new HyPhyFELViewer(this, { theme });
        const src = this.getAttribute('src');
        if (src) {
          fetch(src)
            .then(res => res.json())
            .then(json => this.viewer.load(json))
            .catch(err => console.error('Failed to load JSON from src:', err));
        }
      }
    }

    load(json) {
      this.viewer?.load(json);
    }

    selectSite(idx) {
      this.viewer?.selectSite(idx);
    }

    setTheme(theme) {
      this.viewer?.setTheme(theme);
    }
  }

  customElements.define('hyphy-fel-viewer', HyPhyFELViewerElement);
}
