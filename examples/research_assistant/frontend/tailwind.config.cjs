/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Outfit', 'sans-serif'],
            },
            colors: {
                background: '#0a0a0c',
                sidebar: '#111114',
                glass: {
                    100: 'rgba(255, 255, 255, 0.03)',
                    200: 'rgba(255, 255, 255, 0.05)',
                    300: 'rgba(255, 255, 255, 0.08)',
                    border: 'rgba(255, 255, 255, 0.08)',
                },
                neon: {
                    blue: '#3b82f6',
                    purple: '#a855f7',
                    amber: '#f59e0b',
                }
            },
            animation: {
                'fade-in-up': 'fadeInUp 0.5s ease-out forwards',
                'pulse-glow': 'pulseGlow 2s infinite',
                'shimmer': 'shimmer 2s linear infinite',
            },
            keyframes: {
                fadeInUp: {
                    '0%': { opacity: '0', transform: 'translateY(10px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                pulseGlow: {
                    '0%, 100%': { opacity: '1', boxShadow: '0 0 0px rgba(59, 130, 246, 0)' },
                    '50%': { opacity: '0.8', boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)' },
                },
                shimmer: {
                    '0%': { backgroundPosition: '-1000px 0' },
                    '100%': { backgroundPosition: '1000px 0' },
                }
            }
        },
    },
    plugins: [
        require('@tailwindcss/typography'),
        require('tailwind-scrollbar')({ nocompatible: true }),
    ],
}
