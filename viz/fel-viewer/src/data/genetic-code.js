// Genetic code tables, biophysical chemistry palettes, and Dieter Rams / Braun industrial color schemes

export const CODON_TABLE = {
  // Phenylalanine
  'TTT': 'F', 'TTC': 'F',
  // Leucine
  'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
  // Isoleucine
  'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
  // Methionine (Start)
  'ATG': 'M',
  // Valine
  'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
  // Serine
  'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
  // Proline
  'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
  // Threonine
  'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
  // Alanine
  'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
  // Tyrosine
  'TAT': 'Y', 'TAC': 'Y',
  // Stop
  'TAA': '*', 'TAG': '*', 'TGA': '*',
  // Histidine
  'CAT': 'H', 'CAC': 'H',
  // Glutamine
  'CAA': 'Q', 'CAG': 'Q',
  // Asparagine
  'AAT': 'N', 'AAC': 'N',
  // Lysine
  'AAA': 'K', 'AAG': 'K',
  // Aspartate
  'GAT': 'D', 'GAC': 'D',
  // Glutamate
  'GAA': 'E', 'GAG': 'E',
  // Cysteine
  'TGT': 'C', 'TGC': 'C',
  // Tryptophan
  'TGG': 'W',
  // Arginine
  'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
  // Glycine
  'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
};

export function translateCodon(codon) {
  if (!codon || codon.length !== 3) return '-';
  const clean = codon.toUpperCase().replace(/U/g, 'T');
  return CODON_TABLE[clean] || '?';
}

// Dieter Rams / Braun Industrial Architectural Palette
export const RAMS_COLORS = {
  chassisBg: '#F2F2EF',       // Warm matte off-white chassis
  panelBg: '#FFFFFF',         // Crisp flat white panel
  hairline: '#D1D1CB',        // Precision 1px structural line
  gridLine: '#EAEAE6',        // Subtle cell grid
  ink: '#1A1A18',             // Deep neutral charcoal ink
  stone: '#767672',           // Secondary warm stone label
  dotMuted: '#B8B8B2',        // Muted consensus dots
  signalOrange: '#EB5E28',    // Iconic Braun Signal Orange (Diversifying / Focus)
  industrialCobalt: '#1E40AF',// Technical Cobalt Blue (Purifying)
  amberOchre: '#B45309'       // Precision Amber (Perplexity ⚡)
};

// Calibrated Matte Zappo Palette (Dieter Rams aesthetic: desaturated, tactile industrial tones)
export const ZAPPO_COLORS = {
  // Aliphatic / hydrophobic (matte stone chalk)
  'A': '#E2DDD9', 'I': '#D6CFCA', 'L': '#D6CFCA', 'M': '#D6CFCA', 'V': '#D6CFCA',
  // Aromatic (warm raw sienna)
  'F': '#D97706', 'W': '#B45309', 'Y': '#D97706',
  // Positive / basic (Braun technical cobalt)
  'K': '#1E40AF', 'R': '#1E40AF', 'H': '#3B82F6',
  // Negative / acidic (Braun matte vermilion)
  'D': '#DC2626', 'E': '#DC2626',
  // Conformationally special (warm burnt umber)
  'P': '#9A3412', 'G': '#9A3412',
  // Cysteine (matte brass)
  'C': '#CA8A04',
  // Hydrophilic / polar (slate graphite)
  'S': '#4B5563', 'T': '#4B5563', 'N': '#4B5563', 'Q': '#4B5563',
  // Stop / Gap / Unknown
  '*': '#6B7280', '-': '#E5E7EB', '?': '#D1D5DB'
};

export const NUCLEOTIDE_COLORS = {
  'A': '#1E40AF', // Cobalt
  'C': '#047857', // Sage green
  'G': '#EB5E28', // Braun Orange
  'T': '#DC2626', // Vermilion
  'U': '#DC2626',
  '-': '#D1D5DB',
  'N': '#9CA3AF'
};

// Selection Palette (Disciplined Braun signal colors on neutral ground)
export const SELECTION_COLORS = {
  'diversifying': {
    fill: '#EB5E28',                           // Iconic Braun Orange
    fillLight: 'rgba(235, 94, 40, 0.12)',
    text: '#EB5E28',
    border: '#D9480F',
    label: '▲ Diversifying'
  },
  'purifying': {
    fill: '#1E40AF',                           // Industrial Cobalt
    fillLight: 'rgba(30, 64, 175, 0.10)',
    text: '#1E40AF',
    border: '#1E3A8A',
    label: '▼ Purifying'
  },
  'neutral': {
    fill: '#9CA3AF',
    fillLight: 'rgba(156, 163, 175, 0.06)',
    text: '#6B7280',
    border: '#D1D5DB',
    label: '-'
  }
};

// Paradigm 2 Rate Sparkline: ▰▰▰:▰▰▱ (Machined gauge blocks)
export function getRateSparkGlyphs(alpha, beta) {
  function getSegs(v) {
    if (v == null || isNaN(v) || v <= 1e-4) return 0;
    if (v < 0.5) return 1;
    if (v < 3.0) return 2;
    return 3;
  }
  const na = getSegs(alpha);
  const nb = getSegs(beta);
  const sa = '■'.repeat(na) + '□'.repeat(3 - na);
  const sb = '■'.repeat(nb) + '□'.repeat(3 - nb);
  return { sa, sb, na, nb, text: `${sa}:${sb}` };
}
