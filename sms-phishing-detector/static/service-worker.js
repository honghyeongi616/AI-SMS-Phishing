const CACHE_NAME = "sms-detector-cache-v1";
const urlsToCache = [
    "/",
    "/logs",
    "/score",
    "/static/manifest.json",
    "/static/icon-192.png",
    "/static/icon-512.png"
  ];

self.addEventListener("install", function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener("fetch", function(event) {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});

self.addEventListener("activate", function(event) {
    event.waitUntil(
      caches.keys().then(cacheNames =>
        Promise.all(
          cacheNames.map(function(cacheName) {
            if (cacheName !== CACHE_NAME) {
              return caches.delete(cacheName);
            }
          })
        )
      )
    );
  });
  
  self.addEventListener("install", event => {
    console.log("[ServiceWorker] 설치 완료");
  });
  self.addEventListener("fetch", event => {
    console.log("[ServiceWorker] 요청 캐싱:", event.request.url);
  });
  