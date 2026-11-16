/* GlycoSense — dashboard charts */
(function () {
  "use strict";
  var data = window.dashboardData;
  if (!data || !window.Chart) return;

  var distribution = document.getElementById("distributionChart");
  var models = document.getElementById("modelChart");
  var total = data.distribution.values.reduce(function (a, b) { return a + b; }, 0);

  if (distribution) {
    var hasData = total > 0;
    new Chart(distribution, {
      type: "doughnut",
      data: {
        labels: hasData ? data.distribution.labels : ["No assessments yet"],
        datasets: [{
          data: hasData ? data.distribution.values : [1],
          backgroundColor: hasData ? ["#e2685f", "#2aa98a"] : ["#e9eff7"],
          borderWidth: 0,
          hoverOffset: hasData ? 4 : 0,
          spacing: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "73%",
        plugins: {
          legend: {
            position: "bottom",
            labels: { padding: 16, font: { size: 9.5 } }
          },
          tooltip: {
            enabled: hasData,
            callbacks: {
              label: function (context) {
                var value = context.parsed;
                var share = total ? Math.round((value / total) * 100) : 0;
                return " " + context.label + ": " + value + " (" + share + "%)";
              }
            }
          }
        }
      }
    });
  }

  if (models) {
    new Chart(models, {
      type: "bar",
      data: {
        labels: data.models.labels,
        datasets: [
          {
            label: "Conventional (8 features)",
            data: data.models.baseline,
            backgroundColor: "#dbe6f2",
            hoverBackgroundColor: "#c9d9ea",
            borderRadius: 5,
            maxBarThickness: 26
          },
          {
            label: "AFO-optimized",
            data: data.models.optimized,
            backgroundColor: "#2f8bf5",
            hoverBackgroundColor: "#1f6fe5",
            borderRadius: 5,
            maxBarThickness: 26
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
          x: {
            grid: { display: false },
            ticks: { font: { size: 9.5 }, color: "#6b7f99" }
          },
          y: {
            min: 0,
            max: 1,
            grid: { color: "#eef3f9", drawTicks: false },
            border: { display: false },
            ticks: {
              stepSize: 0.2,
              padding: 8,
              callback: function (value) { return Math.round(value * 100) + "%"; }
            }
          }
        }
      }
    });
  }
})();
