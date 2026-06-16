import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      backdropBlur: { '2xl': '40px' },
      colors: {
        glass: 'rgba(255,255,255,0.08)',
      },
    },
  },
  plugins: [],
}

export default config
