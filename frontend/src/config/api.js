// Set VITE_API_BASE_URL to the deployed backend origin (without /api).
// When unset, requests use the relative /api path and Vite proxies it locally.
const backendOrigin = (import.meta.env.VITE_API_BASE_URL || "").trim().replace(/\/$/, "");

export const API_BASE_URL = backendOrigin ? `${backendOrigin}/api` : "/api";

export function apiAssetUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  if (path.startsWith("/api/")) {
    return backendOrigin ? `${backendOrigin}${path}` : path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}
