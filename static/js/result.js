/* GlycoSense — result probability gauge */
(function () {
  "use strict";
  var data = window.resultData;
  var canvas = document.getElementById("gaugeChart");
  if (!data || !canvas || !window.Chart) return;

  var colors = {
    high: "#df4b4b",
    moderate: "#e0a13a",
    low: "#0f9f74"
  };
  var color = colors[data.risk] || "#2f8bf5";
  var value = Math.max(0, Math.min(100, data.probability));

  new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Diabetes probability", "Remaining"],
      datasets: [{
        data: [value, 100 - value],
        backgroundColor: [color, "#eef3f9"],
        borderWidth: 0,
        hoverOffset: 0,
        spacing: 0,
        circumference: 360
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "79%",
      rotation: -90,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      }
    }
  });
})();
