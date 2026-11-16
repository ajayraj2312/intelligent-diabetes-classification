/* GlycoSense — model performance charts */
(function () {
  "use strict";
  var data = window.performanceData;
  if (!data || !window.Chart) return;

  var colors = window.Glyco.chartColors;
  var palette = [colors.blue, colors.teal, colors.amber, colors.violet];

  var accuracyCanvas = document.getElementById("accuracyChart");
  if (accuracyCanvas) {
    new Chart(accuracyCanvas, {
      type: "bar",
      data: {
        labels: data.models.labels,
        datasets: [
          {
            label: "Conventional",
            data: data.models.baseline,
            backgroundColor: "#dbe6f2",
            hoverBackgroundColor: "#c9d9ea",
            borderRadius: 5,
            maxBarThickness: 24
          },
          {
            label: "AFO-optimized",
            data: data.models.optimized,
            backgroundColor: colors.blue,
            hoverBackgroundColor: "#1f6fe5",
            borderRadius: 5,
            maxBarThickness: 24
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 9.5 } } },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " " + context.dataset.label + ": " + (context.parsed.y * 100).toFixed(1) + "%";
              }
            }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 9 }, maxRotation: 0 } },
          y: {
            min: 0,
            max: 1,
            grid: { color: "#eef3f9" },
            border: { display: false },
            ticks: {
              stepSize: 0.2,
              callback: function (value) { return Math.round(value * 100) + "%"; }
            }
          }
        }
      }
    });
  }

  var radarCanvas = document.getElementById("radarChart");
  if (radarCanvas) {
    new Chart(radarCanvas, {
      type: "radar",
      data: {
        labels: data.radar.labels,
        datasets: data.radar.series.map(function (series, index) {
          var color = palette[index % palette.length];
          return {
            label: series.label,
            data: series.values,
            borderColor: color,
            backgroundColor: color + "14",
            borderWidth: 1.7,
            pointBackgroundColor: color,
            pointRadius: 2,
            pointHoverRadius: 3.5
          };
        })
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 8.5 }, padding: 10 } },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " " + context.dataset.label + ": " + (context.parsed.r * 100).toFixed(1) + "%";
              }
            }
          }
        },
        scales: {
          r: {
            min: 0,
            max: 1,
            ticks: { display: false, stepSize: 0.2 },
            grid: { color: "#e9f0f8", circular: false },
            angleLines: { color: "#edf2f9" },
            pointLabels: { font: { size: 8.5 }, color: "#6b7f99" }
          }
        }
      }
    });
  }

  var rocCanvas = document.getElementById("rocChart");
  if (rocCanvas) {
    var datasets = data.roc.map(function (series, index) {
      var color = palette[index % palette.length];
      return {
        label: series.label + " (AUC " + series.auc.toFixed(3) + ")",
        data: series.fpr.map(function (x, i) { return { x: x, y: series.tpr[i] }; }),
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.8,
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.05
      };
    });
    datasets.push({
      label: "Chance",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: "#c6d3e2",
      borderDash: [4, 4],
      borderWidth: 1,
      pointRadius: 0
    });

    new Chart(rocCanvas, {
      type: "line",
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 8.2 }, padding: 9 } },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " " + context.dataset.label + " · FPR " +
                  (context.parsed.x * 100).toFixed(0) + "% / TPR " +
                  (context.parsed.y * 100).toFixed(0) + "%";
              }
            }
          }
        },
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "False positive rate", font: { size: 8.5 }, color: "#93a5ba" },
            grid: { color: "#eef3f9" },
            border: { display: false },
            ticks: { stepSize: 0.25, callback: function (value) { return Math.round(value * 100) + "%"; }, font: { size: 8.5 } }
          },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: "True positive rate", font: { size: 8.5 }, color: "#93a5ba" },
            grid: { color: "#eef3f9" },
            border: { display: false },
            ticks: { stepSize: 0.25, callback: function (value) { return Math.round(value * 100) + "%"; }, font: { size: 8.5 } }
          }
        }
      }
    });
  }
})();
