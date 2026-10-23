/**
 * FreshTrack AI — Main Application Controller
 */
import { api } from './api.js';
import { CameraHandler } from './camera.js';
import { charts } from './charts.js';

class FreshTrackApp {
  constructor() {
    this.currentView = 'scan';
    this.cameraHandler = null;
    this.lastScanResult = null;
    this.inventoryFilters = {
      status: 'active',
      location: '',
      search: '',
      sort_by: 'expiry',
    };
  }

  async init() {
    console.log('[FreshTrack AI] Initializing app...');
    this.initNavigation();
    this.initScanner();
    this.initInventoryControls();
    this.initModals();
    await this.checkSystemStatus();
    await this.loadSampleButtons();
    await this.loadInventory();
  }

  // Toast System
  showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // Navigation
  initNavigation() {
    const desktopBtns = document.querySelectorAll('.nav-btn');
    const mobileBtns = document.querySelectorAll('.mobile-nav-btn');

    const switchView = (targetView) => {
      this.currentView = targetView;
      // Update buttons
      desktopBtns.forEach((b) => b.classList.toggle('active', b.dataset.view === targetView));
      mobileBtns.forEach((b) => b.classList.toggle('active', b.dataset.view === targetView));

      // Update sections
      document.querySelectorAll('.view-section').forEach((sec) => {
        sec.classList.toggle('active', sec.id === `view-${targetView}`);
      });

      // Trigger view-specific loads
      if (targetView === 'dashboard') {
        this.loadDashboard();
      } else if (targetView === 'inventory') {
        this.loadInventory();
      } else if (targetView === 'storage') {
        this.loadStorageEncyclopedia();
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    desktopBtns.forEach((b) => b.addEventListener('click', () => switchView(b.dataset.view)));
    mobileBtns.forEach((b) => b.addEventListener('click', () => switchView(b.dataset.view)));
  }

  // Check System Status & AI Mode
  async checkSystemStatus() {
    try {
      const status = await api.getSystemStatus();
      const statusPill = document.getElementById('model-status-pill');
      const statusText = document.getElementById('model-status-text');
      const banner = document.getElementById('demo-mode-banner');
      const bannerText = document.getElementById('demo-mode-banner-text');

      if (status.model_status === 'loaded') {
        if (status.ai_mode === 'pretrained_onnx_mobilenet' || (status.produce_model && status.produce_model.includes('MobileNetV2'))) {
          if (statusPill) statusPill.className = 'model-status-pill pytorch';
          if (statusText) statusText.textContent = 'MobileNetV2 AI Active';
        } else if (status.ai_mode === 'pytorch_trained' || status.is_custom_model_loaded) {
          if (statusPill) statusPill.className = 'model-status-pill pytorch';
          if (statusText) statusText.textContent = 'PyTorch AI Active';
        } else {
          if (statusPill) statusPill.className = 'model-status-pill pytorch';
          if (statusText) statusText.textContent = 'AI Model Active';
        }
        if (banner) banner.style.display = 'none';
      } else {
        if (statusPill) statusPill.className = 'model-status-pill unavailable';
        if (statusText) statusText.textContent = 'Model Unavailable';
        if (banner) {
          banner.style.display = 'flex';
          if (bannerText) {
            bannerText.innerHTML = `<strong>AI Model Unavailable:</strong> ${status.model_load_error || 'Model weights could not be loaded'}. Genuine AI inference is currently paused.`;
          }
        }
      }
    } catch (err) {
      console.warn('Could not retrieve system status', err);
      const statusPill = document.getElementById('model-status-pill');
      const statusText = document.getElementById('model-status-text');
      if (statusPill) statusPill.className = 'model-status-pill unavailable';
      if (statusText) statusText.textContent = 'Model Unavailable';
    }
  }

  // Scanner Setup
  initScanner() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const cameraInput = document.getElementById('camera-input');
    const liveCamBtn = document.getElementById('btn-live-camera');
    const cameraModal = document.getElementById('camera-modal');
    const videoElem = document.getElementById('webcam-video');
    const snapBtn = document.getElementById('btn-snap-photo');
    const closeCamBtn = document.getElementById('btn-close-camera');

    this.cameraHandler = new CameraHandler({
      onImageSelected: (file) => this.handleScan(file),
      videoElement: videoElem,
      modalElement: cameraModal,
    });

    this.cameraHandler.initDropzone(dropzone, fileInput);

    if (cameraInput) {
      cameraInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) this.handleScan(file);
      });
    }

    if (liveCamBtn) {
      liveCamBtn.addEventListener('click', async () => {
        try {
          await this.cameraHandler.startLiveCamera();
        } catch (err) {
          this.showToast(err.message, 'error');
        }
      });
    }

    if (snapBtn) {
      snapBtn.addEventListener('click', async () => {
        const file = await this.cameraHandler.captureSnapshot();
        if (file) this.handleScan(file);
      });
    }

    if (closeCamBtn) {
      closeCamBtn.addEventListener('click', () => {
        this.cameraHandler.stopLiveCamera();
      });
    }
  }

  // Sample Buttons
  async loadSampleButtons() {
    const container = document.getElementById('sample-buttons-container');
    if (!container) return;

    try {
      const samples = await api.getSampleImages();
      container.innerHTML = '';
      if (!samples || samples.length === 0) {
        // Fallback default samples
        const defaultSamples = [
          { id: 'fresh_apple', name: '🍎 Fresh Apple' },
          { id: 'spotted_banana', name: '🍌 Spotted Banana' },
          { id: 'fresh_tomato', name: '🍅 Fresh Tomato' },
          { id: 'persimmon', name: '🍊 Persimmon (Kaki)' },
          { id: 'green_lemon', name: '🍋 Green Lemon' },
        ];
        defaultSamples.forEach((s) => {
          const btn = document.createElement('button');
          btn.className = 'sample-btn';
          btn.textContent = s.name;
          btn.addEventListener('click', () => this.handleSampleScan(s.id));
          container.appendChild(btn);
        });
        return;
      }

      samples.forEach((s) => {
        const btn = document.createElement('button');
        btn.className = 'sample-btn';
        let icon = '🥬';
        if (s.id.includes('apple')) icon = '🍎';
        else if (s.id.includes('banana')) icon = '🍌';
        else if (s.id.includes('tomato')) icon = '🍅';
        else if (s.id.includes('persimmon')) icon = '🍊';
        else if (s.id.includes('lemon')) icon = '🍋';
        else if (s.id.includes('carrot')) icon = '🥕';

        btn.textContent = `${icon} ${s.name}`;
        btn.addEventListener('click', () => this.handleSampleScan(s.id));
        container.appendChild(btn);
      });
    } catch (err) {
      console.warn('Failed to load sample buttons', err);
    }
  }

  async handleSampleScan(sampleName) {
    this.setScanningState(true);
    try {
      const result = await api.scanSample(sampleName);
      this.displayScanResults(result);
      this.showToast(`Analyzed ${result.produce}!`, 'success');
    } catch (err) {
      this.showToast(err.message, 'error');
    } finally {
      this.setScanningState(false);
    }
  }

  async handleScan(file) {
    this.setScanningState(true);
    try {
      const result = await api.scanImage(file);
      this.displayScanResults(result);
      this.showToast(`Identified: ${result.produce}`, 'success');
    } catch (err) {
      this.showToast(err.message, 'error');
    } finally {
      this.setScanningState(false);
    }
  }

  setScanningState(isScanning) {
    const loadingElem = document.getElementById('scan-loading');
    const resultsContainer = document.getElementById('results-section');
    if (loadingElem) loadingElem.style.display = isScanning ? 'flex' : 'none';
    if (resultsContainer && isScanning) resultsContainer.style.display = 'none';
  }

  displayScanResults(result) {
    this.lastScanResult = result;
    const resultsContainer = document.getElementById('results-section');
    if (!resultsContainer) return;

    // Image
    const imgElem = document.getElementById('result-img');
    if (imgElem) imgElem.src = result.image_url;

    // Produce Name & Category
    const produceNameElem = document.getElementById('result-produce-name');
    if (produceNameElem) produceNameElem.textContent = result.produce;

    const categoryBadge = document.getElementById('result-category-badge');
    if (categoryBadge) categoryBadge.textContent = result.category;

    // Freshness Badge
    const freshnessBadge = document.getElementById('result-freshness-badge');
    if (freshnessBadge) {
      freshnessBadge.textContent = result.freshness;
      if (result.freshness === 'Fresh') {
        freshnessBadge.className = 'badge badge-fresh';
      } else if (result.freshness.includes('Fair')) {
        freshnessBadge.className = 'badge badge-fair';
      } else {
        freshnessBadge.className = 'badge badge-spoiled';
      }
    }

    // Top Produce Predictions
    const topProduceList = document.getElementById('top-produce-list');
    if (topProduceList && result.top_produce_predictions) {
      topProduceList.innerHTML = '';
      result.top_produce_predictions.forEach((p, idx) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
          <div class="prob-label-row">
            <span>${p.label}</span>
            <span>${p.percentage}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${idx > 0 ? 'alt' : ''}" style="width: ${p.percentage}%"></div>
          </div>
        `;
        topProduceList.appendChild(item);
      });
    }

    // Top Freshness / Surface Discoloration Observations
    const topFreshnessList = document.getElementById('top-freshness-list');
    if (topFreshnessList && result.top_freshness_predictions) {
      topFreshnessList.innerHTML = '';
      result.top_freshness_predictions.forEach((p, idx) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
          <div class="prob-label-row">
            <span>${p.label}</span>
            <span>${p.percentage}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${idx > 0 ? 'alt' : ''}" style="width: ${p.percentage}%"></div>
          </div>
        `;
        topFreshnessList.appendChild(item);
      });
    }

    const freshNotesElem = document.getElementById('result-freshness-notes');
    if (freshNotesElem) {
      freshNotesElem.textContent = result.freshness_notes || '';
    }

    // Gauges: Confidence, Surface Quality & Uncertainty
    const confVal = document.getElementById('result-confidence-val');
    if (confVal) confVal.textContent = `${result.confidence_pct}%`;

    const freshMetricVal = document.getElementById('result-freshness-metric-val');
    if (freshMetricVal) freshMetricVal.textContent = result.freshness;

    const uncertVal = document.getElementById('result-uncertainty-val');
    const uncertHint = document.getElementById('result-uncertainty-hint');
    if (uncertVal) {
      uncertVal.textContent = result.uncertainty_score;
      if (result.is_uncertain) {
        uncertVal.style.color = '#ef4444';
        if (uncertHint) uncertHint.textContent = 'High visual ambiguity';
      } else {
        uncertVal.style.color = '#10b981';
        if (uncertHint) uncertHint.textContent = 'Calibrated low uncertainty';
      }
    }

    // Storage Advice
    const rec = result.storage_recommendation;
    const locElem = document.getElementById('result-storage-loc');
    if (locElem) locElem.textContent = rec.location;

    const ethyleneElem = document.getElementById('result-ethylene-info');
    if (ethyleneElem) ethyleneElem.textContent = rec.ethylene_status;

    const tipsList = document.getElementById('result-tips-list');
    if (tipsList && rec.tips) {
      tipsList.innerHTML = '';
      rec.tips.forEach((tip) => {
        const li = document.createElement('li');
        li.textContent = tip;
        tipsList.appendChild(li);
      });
    }

    // Disclaimer
    const disclaimerElem = document.getElementById('result-disclaimer');
    if (disclaimerElem) disclaimerElem.textContent = result.safety_disclaimer;

    // Notice regarding Demo vs Trained Model
    const noticeElem = document.getElementById('result-model-notice');
    if (noticeElem) {
      noticeElem.textContent = result.model_notice;
    }

    // Low confidence / uncertainty / verification warning banner
    const warningBanner = document.getElementById('result-warning-banner');
    const warningText = document.getElementById('result-warning-text');
    if (warningBanner && warningText) {
      if (result.confidence_pct < 35 || result.produce.includes('Unknown') || result.produce.includes('Ambiguous')) {
        warningText.innerHTML = `<strong>Unknown / Ambiguous Produce:</strong> Model confidence (${result.confidence_pct}%) is below the 35% threshold. Please center the produce with good lighting.`;
        warningBanner.style.background = '#fef2f2';
        warningBanner.style.borderColor = '#f87171';
        warningBanner.style.color = '#991b1b';
        warningBanner.style.display = 'flex';
      } else if (result.confidence_pct < 70) {
        warningText.innerHTML = `<strong>Possible match — please verify:</strong> Neural confidence is ${result.confidence_pct}% (between 35% and 70%). Softmax reflects relative distribution among candidate classes; please verify visual characteristics.`;
        warningBanner.style.background = '#fffbeb';
        warningBanner.style.borderColor = '#f59e0b';
        warningBanner.style.color = '#92400e';
        warningBanner.style.display = 'flex';
      } else if (result.warning_notice) {
        warningText.innerHTML = `<strong>Attention:</strong> ${result.warning_notice}`;
        warningBanner.style.background = '#fffbeb';
        warningBanner.style.borderColor = '#f59e0b';
        warningBanner.style.color = '#92400e';
        warningBanner.style.display = 'flex';
      } else {
        warningBanner.style.display = 'none';
      }
    }

    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Inventory Management
  initInventoryControls() {
    const searchInput = document.getElementById('inv-search-input');
    const locationSelect = document.getElementById('inv-location-filter');
    const sortSelect = document.getElementById('inv-sort-filter');
    const filterPills = document.querySelectorAll('.pill-btn');
    const addManualBtn = document.getElementById('btn-add-manual');
    const saveFromScanBtn = document.getElementById('btn-save-from-scan');
    const exportCsvBtn = document.getElementById('btn-export-csv');
    const exportJsonBtn = document.getElementById('btn-export-json');

    if (searchInput) {
      let debounceTimer;
      searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          this.inventoryFilters.search = e.target.value.trim();
          this.loadInventory();
        }, 300);
      });
    }

    if (locationSelect) {
      locationSelect.addEventListener('change', (e) => {
        this.inventoryFilters.location = e.target.value;
        this.loadInventory();
      });
    }

    if (sortSelect) {
      sortSelect.addEventListener('change', (e) => {
        this.inventoryFilters.sort_by = e.target.value;
        this.loadInventory();
      });
    }

    filterPills.forEach((pill) => {
      pill.addEventListener('click', () => {
        filterPills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        this.inventoryFilters.status = pill.dataset.status;
        this.loadInventory();
      });
    });

    if (addManualBtn) {
      addManualBtn.addEventListener('click', () => this.openInventoryModal());
    }

    if (saveFromScanBtn) {
      saveFromScanBtn.addEventListener('click', () => {
        if (!this.lastScanResult) return;
        this.openInventoryModalFromScan(this.lastScanResult);
      });
    }

    if (exportCsvBtn) {
      exportCsvBtn.addEventListener('click', () => {
        window.open('/api/inventory/export?format=csv', '_blank');
      });
    }

    if (exportJsonBtn) {
      exportJsonBtn.addEventListener('click', () => {
        window.open('/api/inventory/export?format=json', '_blank');
      });
    }
  }

  async loadInventory() {
    const grid = document.getElementById('inventory-grid');
    if (!grid) return;

    try {
      const items = await api.getInventory(this.inventoryFilters);
      grid.innerHTML = '';

      if (!items || items.length === 0) {
        grid.innerHTML = `
          <div style="grid-column: 1/-1; text-align: center; padding: 3rem 1rem; color: var(--text-muted);">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🥗</div>
            <p style="font-weight: 600; font-size: 1.1rem; color: var(--text-main);">No items found</p>
            <p style="font-size: 0.9rem;">No produce items match your current filter criteria.</p>
          </div>
        `;
        return;
      }

      items.forEach((it) => {
        const card = document.createElement('div');
        card.className = 'item-card';

        let badgeClass = 'badge-fresh';
        if (it.urgency_status === 'Expiring Soon') badgeClass = 'badge-fair';
        else if (it.urgency_status === 'Expired' || it.urgency_status === 'Discarded') badgeClass = 'badge-spoiled';

        let daysText = `${it.days_remaining} days left`;
        if (it.days_remaining === 0) daysText = 'Expires today!';
        else if (it.days_remaining < 0) daysText = `Expired ${Math.abs(it.days_remaining)} days ago`;

        card.innerHTML = `
          <div class="item-header">
            <div>
              <div class="item-name">${it.produce_name}</div>
              <div class="badge badge-category" style="margin-top: 4px; display: inline-block;">${it.category}</div>
            </div>
            <span class="badge ${badgeClass}">${it.urgency_status}</span>
          </div>
          <div class="item-meta">
            <div><strong>Quantity:</strong> ${it.quantity} ${it.unit}</div>
            <div><strong>Storage:</strong> ${it.storage_location}</div>
            <div><strong>Estimated Expiry:</strong> ${it.estimated_expiry_date} (${daysText})</div>
            ${it.notes ? `<div style="font-style: italic;">"${it.notes}"</div>` : ''}
          </div>
          <div class="item-actions">
            ${
              it.status === 'active'
                ? `
              <button class="btn btn-success btn-sm btn-consume" data-id="${it.id}">
                ✓ Consumed
              </button>
              <button class="btn btn-danger btn-sm btn-discard" data-id="${it.id}">
                ✕ Discarded
              </button>
            `
                : `<span style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600;">Status: ${it.status.toUpperCase()}</span>`
            }
            <button class="btn btn-outline btn-sm btn-delete" data-id="${it.id}" style="margin-left: auto;">
              🗑
            </button>
          </div>
        `;

        // Action Bindings
        const consumeBtn = card.querySelector('.btn-consume');
        if (consumeBtn) {
          consumeBtn.addEventListener('click', async () => {
            try {
              await api.recordAction(it.id, 'consumed');
              this.showToast(`Marked ${it.produce_name} as Consumed! Saved food waste.`, 'success');
              this.loadInventory();
            } catch (err) {
              this.showToast(err.message, 'error');
            }
          });
        }

        const discardBtn = card.querySelector('.btn-discard');
        if (discardBtn) {
          discardBtn.addEventListener('click', async () => {
            try {
              await api.recordAction(it.id, 'discarded');
              this.showToast(`Logged ${it.produce_name} as Discarded.`, 'info');
              this.loadInventory();
            } catch (err) {
              this.showToast(err.message, 'error');
            }
          });
        }

        const deleteBtn = card.querySelector('.btn-delete');
        if (deleteBtn) {
          deleteBtn.addEventListener('click', async () => {
            if (confirm(`Are you sure you want to delete ${it.produce_name} from inventory?`)) {
              try {
                await api.deleteInventoryItem(it.id);
                this.showToast('Item deleted', 'info');
                this.loadInventory();
              } catch (err) {
                this.showToast(err.message, 'error');
              }
            }
          });
        }

        grid.appendChild(card);
      });
    } catch (err) {
      console.error(err);
      this.showToast('Failed to load inventory items', 'error');
    }
  }

  // Modals
  initModals() {
    const modal = document.getElementById('inventory-modal');
    const closeBtn = document.getElementById('btn-close-inv-modal');
    const form = document.getElementById('inventory-form');

    if (closeBtn && modal) {
      closeBtn.addEventListener('click', () => modal.classList.remove('active'));
    }

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const data = {
          produce_name: form.produce_name.value.trim(),
          category: form.category.value,
          freshness_state: form.freshness_state.value,
          quantity: parseFloat(form.quantity.value) || 1,
          unit: form.unit.value,
          storage_location: form.storage_location.value,
          purchase_date: form.purchase_date.value,
          estimated_expiry_date: form.estimated_expiry_date.value,
          notes: form.notes.value.trim(),
        };

        try {
          await api.createInventoryItem(data);
          this.showToast(`Saved ${data.produce_name} to inventory!`, 'success');
          modal.classList.remove('active');
          form.reset();
          this.loadInventory();
        } catch (err) {
          this.showToast(err.message, 'error');
        }
      });
    }
  }

  openInventoryModal() {
    const modal = document.getElementById('inventory-modal');
    const form = document.getElementById('inventory-form');
    if (!modal || !form) return;

    form.reset();
    const today = new Date().toISOString().split('T')[0];
    const expiry = new Date(Date.now() + 7 * 86400000).toISOString().split('T')[0];
    form.purchase_date.value = today;
    form.estimated_expiry_date.value = expiry;
    modal.classList.add('active');
  }

  openInventoryModalFromScan(scan) {
    const modal = document.getElementById('inventory-modal');
    const form = document.getElementById('inventory-form');
    if (!modal || !form) return;

    form.reset();
    form.produce_name.value = scan.produce;
    form.category.value = scan.category || 'Fruit';
    form.freshness_state.value = scan.freshness;
    form.storage_location.value = scan.storage_recommendation.location;

    const today = new Date();
    form.purchase_date.value = today.toISOString().split('T')[0];

    const shelfDays = scan.storage_recommendation.estimated_shelf_life_days || 5;
    const expiryDate = new Date(today.getTime() + shelfDays * 86400000);
    form.estimated_expiry_date.value = expiryDate.toISOString().split('T')[0];

    form.notes.value = `Scanned via FreshTrack AI (${scan.confidence_pct}% confidence). Recommended: ${scan.storage_recommendation.location}.`;
    modal.classList.add('active');
  }

  // Dashboard
  async loadDashboard() {
    try {
      const data = await api.getDashboardStats();

      const totalElem = document.getElementById('kpi-total-active');
      if (totalElem) totalElem.textContent = data.total_active_items;

      const expiringElem = document.getElementById('kpi-expiring-soon');
      if (expiringElem) expiringElem.textContent = data.expiring_soon_count;

      const preventionElem = document.getElementById('kpi-prevention-rate');
      if (preventionElem) preventionElem.textContent = `${data.waste_prevention_rate_pct}%`;

      const savedElem = document.getElementById('kpi-money-saved');
      if (savedElem) savedElem.textContent = `$${data.estimated_money_saved.toFixed(2)}`;

      // Render Charts
      charts.renderFreshnessChart('chart-freshness', data.freshness_distribution);
      charts.renderStorageChart('chart-storage', data.storage_distribution);

      // Render Recent Scans
      const recentList = document.getElementById('recent-scans-list');
      if (recentList && data.recent_scans) {
        recentList.innerHTML = '';
        if (data.recent_scans.length === 0) {
          recentList.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No scans recorded yet.</td></tr>';
        } else {
          data.recent_scans.forEach((sc) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <img src="${sc.image_url}" style="width: 36px; height: 36px; border-radius: 6px; object-fit: cover;" onerror="this.src='/frontend/assets/samples/fresh_apple.jpg'"/>
                  <strong>${sc.produce_detected}</strong>
                </div>
              </td>
              <td><span class="badge ${sc.freshness_detected === 'Fresh' ? 'badge-fresh' : 'badge-fair'}">${sc.freshness_detected}</span></td>
              <td>${Math.round(sc.confidence_score * 100)}%</td>
              <td style="color: var(--text-muted); font-size: 0.82rem;">${new Date(sc.scanned_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
            `;
            recentList.appendChild(tr);
          });
        }
      }
    } catch (err) {
      console.error(err);
      this.showToast('Failed to load dashboard metrics', 'error');
    }
  }

  // Storage Encyclopedia
  async loadStorageEncyclopedia() {
    const grid = document.getElementById('storage-guides-grid');
    if (!grid) return;

    try {
      const guides = await api.getStorageGuides();
      grid.innerHTML = '';

      guides.forEach((g) => {
        const card = document.createElement('div');
        card.className = 'guide-card';
        card.innerHTML = `
          <div class="guide-name">
            <span>${g.name}</span>
            <span class="guide-loc-badge">${g.optimal_location}</span>
          </div>
          <div class="guide-details">
            <div><strong>Ideal Temp:</strong> ${g.temperature_range}</div>
            <div><strong>Humidity:</strong> ${g.humidity_recommendation}</div>
            <div><strong>Ethylene:</strong> ${g.ethylene_profile}</div>
          </div>
          <div style="background: var(--bg-subtle); padding: 0.75rem; border-radius: 8px; font-size: 0.82rem;">
            <strong>Pro Tip:</strong> ${g.preservation_tips[0] || 'Keep in cool, ventilated area.'}
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">
            <strong>Freezing:</strong> ${g.can_freeze ? 'Yes (' + g.freezing_method + ')' : 'Not recommended (loss of texture)'}
          </div>
        `;
        grid.appendChild(card);
      });
    } catch (err) {
      console.error(err);
    }
  }
}

// Instantiate on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  const app = new FreshTrackApp();
  app.init();
});
