const CACHE_NAME = 'health-triage-cache-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/static/css/styles.css',
  '/static/js/app.js',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap',
  'https://unpkg.com/@splinetool/viewer@1.12.73/build/spline-viewer.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          return response;
        }
        return fetch(event.request);
      }
    )
  );
});
