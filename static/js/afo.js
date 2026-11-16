/* GlycoSense — Archerfish Optimization charts */
(function () {
  "use strict";
  var data = window.afoData;
  if (!data || !window.Chart) return;

  var convergence = document.getElementById("convergenceChart");
  if (convergence) {
    var trace = data.convergence;
    new Chart(convergence, {
      type: "line",
      data: {
        labels: trace.map(function (point) { return point.iteration; }),
        datasets: [
          {
            label: "Best fitness",
            data: trace.map(function (point) { return point.best_fitness; }),
            borderColor: "#2f8bf5",
            backgroundColor: "rgba(47,139,245,.10)",
            borderWidth: 2.2,
            pointRadius: 2,
            pointHoverRadius: 4,
            fill: true,
            tension: 0.28
          },
          {
            label: "School mean",
            data: trace.map(function (point) { return point.mean_fitness; }),
            borderColor: "#9fb2c7",
            borderWidth: 1.4,
            pointRadius: 0,
            borderDash: [3, 3],
            tension: 0.25
          },
          {
            label: "Conventional baseline",
            data: trace.map(function () { return data.baseline_fitness; }),
            borderColor: "#e0a13a",
            borderWidth: 1.2,
            pointRadius: 0,
            borderDash: [6, 4]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " " + context.dataset.label + ": " + context.parsed.y.toFixed(4);
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: "Iteration", font: { size: 8.5 }, color: "#93a5ba" },
            grid: { display: false },
            ticks: { font: { size: 8.5 }, maxTicksLimit: 8 }
          },
          y: {
            title: { display: true, text: "Fitness J (lower is better)", font: { size: 8.5 }, color: "#93a5ba" },
            grid: { color: "#eef3f9" },
            border: { display: false },
            ticks: {
              font: { size: 8.5 },
              callback: function (value) { return value.toFixed(2); }
            }
          }
        }
      }
    });
  }

  var frequency = document.getElementById("frequencyChart");
  if (frequency) {
    var features = data.features;
    var values = features.map(function (feature) {
      return Math.round(data.frequency[feature] * 100);
    });
    var backgrounds = features.map(function (feature) {
      return data.selected.indexOf(feature) !== -1 ? "#2f8bf5" : "#d5dfeb";
    });

    new Chart(frequency, {
      type: "bar",
      data: {
        labels: features.map(function (feature, index) { return data.labels[index]; }),
        datasets: [{
          label: "Selection frequency",
          data: values,
          backgroundColor: backgrounds,
          borderRadius: 4,
          maxBarThickness: 14
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (context) {
                return " Included in " + context.parsed.x + "% of the final school";
              }
            }
          }
        },
        scales: {
          x: {
            min: 0,
            max: 100,
            grid: { color: "#eef3f9" },
            border: { display: false },
            ticks: {
              font: { size: 8 },
              callback: function (value) { return value + "%"; }
            }
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 8.5 }, color: "#4f6882" }
          }
        }
      }
    });
  }
})();
