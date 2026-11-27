// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Published on GitHub Pages at https://misterekkk.github.io/miter/ (see .github/workflows/docs.yml).
export default defineConfig({
  site: 'https://misterekkk.github.io',
  base: '/miter',
  integrations: [
    starlight({
      title: 'miter',
      description: 'Gradient-boosted decision trees with random oblique splits.',
      logo: { src: './assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/misterekkk/miter' }],
      editLink: { baseUrl: 'https://github.com/misterekkk/miter/edit/main/docs/' },
      customCss: ['@fontsource-variable/inter', '@fontsource-variable/jetbrains-mono', './src/styles/custom.css'],
      sidebar: [
        { label: 'Start', items: ['guides/getting-started'] },
        {
          label: 'Guides',
          items: ['guides/parameters', 'guides/categorical', 'guides/scikit-learn', 'guides/saving', 'guides/tabarena'],
        },
        { label: 'Concepts', items: ['concepts/oblique-splits', 'concepts/determinism'] },
        { label: 'Benchmarks', items: ['benchmarks'] },
        { label: 'Reference', items: ['reference/python'] },
        { label: 'Project', items: ['release-notes'] },
      ],
    }),
  ],
});
