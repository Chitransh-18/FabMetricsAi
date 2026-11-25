/**
 * FabMetrics AI — Progressive Web App (PWA) Service Worker
 * Enables offline shell caching and fast mobile web app installation.
 */

const CACHE_NAME = 'fabmetrics-v4-cache';
const SHELL_ASSETS = [
  '/',
  '/manifest.json',
  '/frontend/index.html',
  '/frontend/404.html',
  '/frontend/themes.js',
  '/frontend/assets/logo.png',
  '/frontend/assets/favicon.png',
  '/frontend/assets/sample_wafer.png'
];

// Install: Cache critical shell assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Pre-caching offline shell assets');
      return cache.addAll(SHELL_ASSETS).catch((err) => {
        console.warn('[ServiceWorker] Some assets could not be pre-cached:', err);
      });
    }).then(() => self.skipWaiting())
  );
});

// Activate: Clean up old cache versions
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing legacy cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: Stale-while-revalidate for static files, Network-first for APIs
self.addEventListener('fetch', (event) => {
  const requestUrl = new URL(event.request.url);

  // For API endpoints and dynamic POST requests, always use network
  if (event.request.method !== 'GET' || requestUrl.pathname.startsWith('/api/') || requestUrl.pathname.startsWith('/predict')) {
    return;
  }

  // Stale-While-Revalidate for UI shell
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      const fetchPromise = fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      }).catch(() => {
        // Fallback for navigation requests when totally offline
        if (event.request.mode === 'navigate') {
          return caches.match('/frontend/index.html');
        }
      });

      return cachedResponse || fetchPromise;
    })
  );
});
