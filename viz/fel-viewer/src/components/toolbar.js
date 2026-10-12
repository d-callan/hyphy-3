// Top Toolbar component for view mode toggling, color scheme selection, site search, and threshold controls

export class Toolbar {
  constructor(container, options = {}) {
    this.container = container;
    this.options = Object.assign({
      onModeChange: null,
      onSchemeChange: null,
      onSiteJump: null,
      onThresholdChange: null,
      onFileLoad: null,
      onSampleSelect: null,
      onThemeToggle: null,
      onAncestorsToggle: null,
      onOverviewToggle: null
    }, options);

    this.el = document.createElement('div');
    this.el.className = 'hyphy-toolbar';
    this.container.appendChild(this.el);

    this.render();
    this.setupEvents();
  }

  render() {
    this.el.innerHTML = `
      <div class="toolbar-group">
        <div class="dataset-dropdown-wrapper">
          <label class="toolbar-label">Dataset:</label>
          <select class="toolbar-select" id="sample-select">
            <option value="cd2_meme">Sample: CD2 (MEME Episodic, 10 taxa)</option>
            <option value="bglobin" selected>Sample: Beta-Globin (FEL Pervasive, 17 taxa)</option>
            <option value="fig4e">Sample: Fig4E Partitioned (FEL, 33 taxa)</option>
            <option value="custom">Load Custom File...</option>
          </select>
          <input type="file" id="file-input" accept=".json,.fna,.fasta,.nex,.nexus" style="display:none;" />
        </div>
      </div>

      <div class="toolbar-divider"></div>

      <div class="toolbar-group">
        <label class="toolbar-label">View:</label>
        <div class="segmented-control" id="mode-control">
          <button class="seg-btn active" data-mode="amino_acid">Amino Acids</button>
          <button class="seg-btn" data-mode="codon">Codons (3nt)</button>
        </div>
      </div>

      <div class="toolbar-group">
        <label class="toolbar-label">Color:</label>
        <select class="toolbar-select" id="scheme-select">
          <option value="difference" selected>Diff from Consensus</option>
          <option value="difference_root">Diff from Root (MRCA)</option>
          <option value="selection">Selection State</option>
          <option value="chemistry">Biophysical Chemistry (Zappo)</option>
        </select>
      </div>

      <div class="toolbar-divider"></div>

      <div class="toolbar-group">
        <label class="toolbar-label">Ancestors:</label>
        <div class="segmented-control" id="ancestors-control">
          <button class="seg-btn active" data-ancestors="off">Extant Only</button>
          <button class="seg-btn" data-ancestors="on">+ Inferred Ancestors</button>
        </div>
      </div>

      <div class="toolbar-divider"></div>

      <div class="toolbar-group">
        <label class="toolbar-label">Jump to Site:</label>
        <div class="site-jump-box">
          <input type="number" id="site-jump-input" class="toolbar-input" min="1" placeholder="Site #" />
          <button class="btn-jump" id="btn-jump">Go</button>
        </div>
      </div>

      <div class="toolbar-group">
        <label class="toolbar-label">p-value Cutoff:</label>
        <select class="toolbar-select" id="threshold-select">
          <option value="0.10" selected>p ≤ 0.10</option>
          <option value="0.05">p ≤ 0.05</option>
          <option value="0.01">p ≤ 0.01</option>
        </select>
      </div>

      <div class="toolbar-divider"></div>

      <div class="toolbar-group">
        <button class="toolbar-btn active" id="btn-toggle-overview" title="Toggle Bird's Eye Overview Panel">☷ Overview</button>
      </div>

      <div class="toolbar-spacer"></div>

      <div class="toolbar-group">
        <button class="toolbar-btn" id="theme-btn" title="Toggle Dark/Light Mode">🌓</button>
      </div>
    `;
  }

  setupEvents() {
    // Dataset Selection
    const sampleSelect = this.el.querySelector('#sample-select');
    const fileInput = this.el.querySelector('#file-input');

    sampleSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (val === 'custom') {
        fileInput.click();
      } else if (this.options.onSampleSelect) {
        this.options.onSampleSelect(val);
      }
    });

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file && this.options.onFileLoad) {
        this.options.onFileLoad(file);
      }
    });

    // Translation Mode
    const modeButtons = this.el.querySelectorAll('#mode-control .seg-btn');
    modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        modeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.getAttribute('data-mode');
        if (this.options.onModeChange) this.options.onModeChange(mode);
      });
    });

    // Color Scheme
    const schemeSelect = this.el.querySelector('#scheme-select');
    schemeSelect.addEventListener('change', (e) => {
      if (this.options.onSchemeChange) this.options.onSchemeChange(e.target.value);
    });

    // Ancestral Nodes Toggle
    const ancButtons = this.el.querySelectorAll('#ancestors-control .seg-btn');
    ancButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        ancButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const show = btn.getAttribute('data-ancestors') === 'on';
        if (this.options.onAncestorsToggle) this.options.onAncestorsToggle(show);
      });
    });

    // Site Jump
    const jumpInput = this.el.querySelector('#site-jump-input');
    const jumpBtn = this.el.querySelector('#btn-jump');
    const doJump = () => {
      const val = parseInt(jumpInput.value, 10);
      if (!isNaN(val) && this.options.onSiteJump) {
        this.options.onSiteJump(val);
      }
    };
    jumpBtn.addEventListener('click', doJump);
    jumpInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') doJump();
    });

    // P-value Threshold
    const threshSelect = this.el.querySelector('#threshold-select');
    threshSelect.addEventListener('change', (e) => {
      const p = parseFloat(e.target.value);
      if (this.options.onThresholdChange) this.options.onThresholdChange(p);
    });

    // Theme Toggle
    const themeBtn = this.el.querySelector('#theme-btn');
    themeBtn.addEventListener('click', () => {
      if (this.options.onThemeToggle) this.options.onThemeToggle();
    });

    // Overview Panel Toggle
    const ovBtn = this.el.querySelector('#btn-toggle-overview');
    if (ovBtn) {
      ovBtn.addEventListener('click', () => {
        ovBtn.classList.toggle('active');
        if (this.options.onOverviewToggle) {
          this.options.onOverviewToggle();
        }
      });
    }
  }

  setDatasetName(val) {
    const sel = this.el.querySelector('#sample-select');
    if (!sel) return;
    for (const opt of sel.options) {
      if (opt.value === val) {
        sel.value = val;
        return;
      }
    }
  }
}
