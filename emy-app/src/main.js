import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './assets/tailwind.css'
import Vue3Geolocation from 'vue3-geolocation';

createApp(App).use(router).use(Vue3Geolocation).mount('#app')

