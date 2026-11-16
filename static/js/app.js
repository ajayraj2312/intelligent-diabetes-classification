/* GlycoSense — shared interface behaviour */
(function () {
  "use strict";

  /* ---------------------------------------------------------------- */
  /* Chart.js global defaults                                          */
  /* ---------------------------------------------------------------- */
  if (window.Chart) {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 10;
    Chart.defaults.color = "#6b7f99";
    Chart.defaults.borderColor = "#eaf0f7";
    Chart.defaults.devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 7;
    Chart.defaults.plugins.legend.labels.boxHeight = 7;
    Chart.defaults.plugins.legend.labels.padding = 14;
    Chart.defaults.plugins.tooltip.backgroundColor = "#0d2440";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.titleFont = { size: 10, weight: "600" };
    Chart.defaults.plugins.tooltip.bodyFont = { size: 10 };
    Chart.defaults.animation.duration = 650;
  }

  window.Glyco = window.Glyco || {};
  window.Glyco.percent = function (value) {
    return Math.round(value * 100) + "%";
  };
  window.Glyco.chartColors = {
    blue: "#2f8bf5",
    teal: "#0f9f8a",
    amber: "#e0a13a",
    violet: "#8b76e8",
    slate: "#94a9c1",
    green: "#0f9f74",
    red: "#df4b4b"
  };

  /* ---------------------------------------------------------------- */
  /* Sidebar                                                          */
  /* ---------------------------------------------------------------- */
  var sidebar = document.getElementById("sidebar");
  var overlay = document.querySelector(".sidebar-overlay");
  var toggle = document.querySelector("[data-sidebar-toggle]");

  function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove("is-open");
    if (overlay) overlay.classList.remove("is-open");
    document.body.style.overflow = "";
  }

  function openSidebar() {
    if (!sidebar) return;
    sidebar.classList.add("is-open");
    if (overlay) overlay.classList.add("is-open");
    document.body.style.overflow = "hidden";
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      if (sidebar.classList.contains("is-open")) closeSidebar();
      else openSidebar();
    });
  }
  if (overlay) overlay.addEventListener("click", closeSidebar);
  document.querySelectorAll(".sidebar .nav-link").forEach(function (link) {
    link.addEventListener("click", function () {
      if (window.innerWidth <= 1024) closeSidebar();
    });
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeSidebar();
  });
  window.addEventListener("resize", function () {
    if (window.innerWidth > 1024) {
      closeSidebar();
      if (sidebar) sidebar.classList.remove("is-open");
    }
  });

  /* ---------------------------------------------------------------- */
  /* Flash messages                                                   */
  /* ---------------------------------------------------------------- */
  document.querySelectorAll("[data-dismiss-flash]").forEach(function (button) {
    button.addEventListener("click", function () {
      var flash = button.closest(".flash");
      if (flash) flash.remove();
    });
  });
  window.setTimeout(function () {
    document.querySelectorAll(".flash").forEach(function (flash) {
      flash.style.transition = "opacity .4s ease, transform .4s ease";
      flash.style.opacity = "0";
      flash.style.transform = "translateY(-4px)";
      window.setTimeout(function () { flash.remove(); }, 420);
    });
  }, 7000);

  /* ---------------------------------------------------------------- */
  /* Form validation & submission state                               */
  /* ---------------------------------------------------------------- */
  function clearClientErrors(form) {
    form.querySelectorAll(".field").forEach(function (field) {
      field.classList.remove("has-error");
      field.querySelectorAll(".field-error").forEach(function (error) {
        error.remove();
      });
      var input = field.querySelector(".form-input");
      if (input) input.setAttribute("aria-invalid", "false");
    });
  }

  function fieldMessage(input) {
    var raw = (input.value || "").trim();
    if (!raw) return "This field is required.";
    var value = Number(raw);
    if (!Number.isFinite(value)) return "Enter a numeric value.";
    var min = input.getAttribute("min");
    var max = input.getAttribute("max");
    if (min !== null && value < Number(min)) return "Value must be at least " + min + ".";
    if (max !== null && value > Number(max)) return "Value must not exceed " + max + ".";
    return "";
  }

  function validateForm(form) {
    var firstInvalid = null;
    form.querySelectorAll("input[required]").forEach(function (input) {
      var field = input.closest(".field");
      if (!field) return;
      var message = fieldMessage(input);
      if (!message) return;

      if (!firstInvalid) firstInvalid = input;
      field.classList.add("has-error");
      input.setAttribute("aria-invalid", "true");
      var hint = field.querySelector(".field-hint");
      var error = document.createElement("p");
      error.className = "field-error client-generated";
      error.textContent = message;
      if (hint) hint.insertAdjacentElement("afterend", error);
      else input.insertAdjacentElement("afterend", error);
    });
    return firstInvalid;
  }

  document.querySelectorAll("form[data-loading-form]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      clearClientErrors(form);

      var invalid = validateForm(form);
      if (invalid) {
        event.preventDefault();
        invalid.focus({ preventScroll: true });
        invalid.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }

      var confirmMessage = form.getAttribute("data-confirm");
      if (confirmMessage && !window.confirm(confirmMessage)) {
        event.preventDefault();
        return;
      }

      form.classList.add("is-loading");
      var submit = form.querySelector('button[type="submit"]');
      if (submit) {
        submit.disabled = true;
        submit.dataset.originalText = submit.innerHTML;
        submit.innerHTML = '<span class="spinner"></span> Processing…';
      }
    });
  });

  document.querySelectorAll('form button[type="reset"]').forEach(function (button) {
    button.addEventListener("click", function () {
      var form = button.closest("form");
      if (!form) return;
      window.setTimeout(function () { clearClientErrors(form); }, 0);
    });
  });

  /* Keep a submit button from being restored by browser back navigation. */
  window.addEventListener("pageshow", function (event) {
    if (!event.persisted) return;
    document.querySelectorAll("form.is-loading").forEach(function (form) {
      form.classList.remove("is-loading");
      var submit = form.querySelector('button[type="submit"]');
      if (submit) {
        submit.disabled = false;
        if (submit.dataset.originalText) submit.innerHTML = submit.dataset.originalText;
      }
    });
  });
})();
