/**
 * FreshTrack AI API Client Module
 */
const API_BASE = window.location.origin;

export const api = {
  // Scans & Inference
  async scanImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/api/scan`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Scan failed' }));
      throw new Error(err.detail || 'Image scanning failed');
    }
    return await res.json();
  },

  async scanSample(sampleName) {
    const res = await fetch(`${API_BASE}/api/scan/sample/${sampleName}`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Sample scan failed' }));
      throw new Error(err.detail || 'Sample scan failed');
    }
    return await res.json();
  },

  async getSampleImages() {
    const res = await fetch(`${API_BASE}/api/sample-images`);
    if (!res.ok) return [];
    return await res.json();
  },

  async getScanHistory(limit = 20) {
    const res = await fetch(`${API_BASE}/api/scan/history?limit=${limit}`);
    if (!res.ok) return [];
    return await res.json();
  },

  async clearScanHistory() {
    const res = await fetch(`${API_BASE}/api/scan/history`, { method: 'DELETE' });
    return await res.json();
  },

  // Inventory
  async getInventory(filters = {}) {
    const params = new URLSearchParams();
    if (filters.status) params.append('status', filters.status);
    if (filters.location) params.append('location', filters.location);
    if (filters.category) params.append('category', filters.category);
    if (filters.search) params.append('search', filters.search);
    if (filters.sort_by) params.append('sort_by', filters.sort_by);

    const res = await fetch(`${API_BASE}/api/inventory?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to load inventory');
    return await res.json();
  },

  async createInventoryItem(data) {
    const res = await fetch(`${API_BASE}/api/inventory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to create item' }));
      throw new Error(err.detail || 'Failed to create inventory item');
    }
    return await res.json();
  },

  async updateInventoryItem(id, data) {
    const res = await fetch(`${API_BASE}/api/inventory/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to update inventory item');
    return await res.json();
  },

  async deleteInventoryItem(id) {
    const res = await fetch(`${API_BASE}/api/inventory/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete item');
    return await res.json();
  },

  async recordAction(id, action, quantity, estimated_value = 0.0) {
    const res = await fetch(`${API_BASE}/api/inventory/${id}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, quantity, estimated_value }),
    });
    if (!res.ok) throw new Error(`Failed to mark item as ${action}`);
    return await res.json();
  },

  // Dashboard & System Status
  async getDashboardStats() {
    const res = await fetch(`${API_BASE}/api/dashboard`);
    if (!res.ok) throw new Error('Failed to load dashboard metrics');
    return await res.json();
  },

  async getSystemStatus() {
    const res = await fetch(`${API_BASE}/api/system-status`);
    if (!res.ok) return { ai_mode: 'demo_heuristic', is_custom_model_loaded: false };
    return await res.json();
  },

  // Storage Guides
  async getStorageGuides() {
    const res = await fetch(`${API_BASE}/api/storage-guides`);
    if (!res.ok) return [];
    return await res.json();
  },
};
