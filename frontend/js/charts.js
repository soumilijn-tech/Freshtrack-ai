/**
 * FreshTrack AI Chart.js Visualizations
 */
let freshnessChartInstance = null;
let storageChartInstance = null;

export const charts = {
  renderFreshnessChart(canvasId, distribution) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === 'undefined') return;

    if (freshnessChartInstance) {
      freshnessChartInstance.destroy();
    }

    const labels = ['Fresh', 'Fair / Consume Soon', 'Overripe / Spoiled'];
    const data = [
      distribution['Fresh'] || 0,
      distribution['Fair / Consume Soon'] || 0,
      distribution['Overripe / Spoiled'] || 0,
    ];

    freshnessChartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [
          {
            data: data,
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderWidth: 2,
            borderColor: '#ffffff',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 12,
              font: { size: 12, family: 'system-ui' },
            },
          },
        },
        cutout: '70%',
      },
    });
  },

  renderStorageChart(canvasId, storageDist) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === 'undefined') return;

    if (storageChartInstance) {
      storageChartInstance.destroy();
    }

    const labels = Object.keys(storageDist);
    const data = Object.values(storageDist);

    storageChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Items',
            data: data,
            backgroundColor: '#059669',
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { precision: 0, font: { family: 'system-ui' } },
            grid: { color: '#f1f5f9' },
          },
          x: {
            ticks: { font: { family: 'system-ui' } },
            grid: { display: false },
          },
        },
      },
    });
  },
};
