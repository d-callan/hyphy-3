import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig(({ mode }) => {
  if (mode === 'lib') {
    return {
      build: {
        lib: {
          entry: resolve(__dirname, 'src/index.js'),
          name: 'HyPhyFELViewer',
          fileName: (format) => `hyphy-fel-viewer.${format === 'es' ? 'js' : 'umd.cjs'}`
        },
        outDir: 'dist',
        sourcemap: true,
        rollupOptions: {
          output: {
            exports: 'named',
            assetFileNames: (assetInfo) => {
              if (assetInfo.name === 'style.css') return 'hyphy-fel-viewer.css';
              return assetInfo.name;
            }
          }
        }
      }
    };
  }

  return {
    build: {
      outDir: 'dist-demo',
      sourcemap: true
    }
  };
});
