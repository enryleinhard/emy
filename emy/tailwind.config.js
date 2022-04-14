const defaultTheme = require("tailwindcss/defaultTheme");

module.exports = {
  purge: [
    "./components/**/*.{vue,js}",
    "./layouts/**/*.vue",
    "./pages/**/*.vue",
    "./plugins/**/*.{js,ts}",
    "./nuxt.config.{js,ts}"
  ],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      fontFamily: {
        // sans: ["Poppins", ...defaultTheme.fontFamily.sans],
        sans: ["Plus Jakarta Sans", ...defaultTheme.fontFamily.sans],
      },
      colors: {
        first: {
          50  : '#456c6d',
          100 : '#3f6768',
          200 : '#386162',
          300 : '#325c5d',
          400 : '#2c5758',
          500 : '#255253',
          600 : '#1f4d4e',
          700 : '#174849',
          800 : '#0f4344',
          900 : '#063e3f',
        },
        second: '#839F9F',
        third: '#C1CFCF',
        fourth: '#E0E7E7',
        fifth: 'FD9FA7',
      }
    },
  },
  variants: {
    extend: {}
  },
};
