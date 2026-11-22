// Progressive enhancement only: every page works without this file.
document.addEventListener("DOMContentLoaded", () => {
  // <input data-filter="table-id">: hide rows that do not contain the typed text.
  for (const input of document.querySelectorAll("input[data-filter]")) {
    const table = document.getElementById(input.dataset.filter);
    if (!table) continue;
    input.addEventListener("input", () => {
      const needle = input.value.trim().toLowerCase();
      for (const row of table.querySelectorAll("tbody tr")) {
        row.hidden = needle !== "" && !row.textContent.toLowerCase().includes(needle);
      }
    });
  }

  // <div data-toggle-group="target-id"><button data-show="kind">: show only items whose data-kind includes it.
  for (const group of document.querySelectorAll("[data-toggle-group]")) {
    const target = document.getElementById(group.dataset.toggleGroup);
    if (!target) continue;
    group.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-show]");
      if (!button) return;
      for (const other of group.querySelectorAll("button[data-show]")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      const show = button.dataset.show;
      for (const item of target.querySelectorAll("[data-kind]")) {
        item.hidden = show !== "all" && !item.dataset.kind.split(" ").includes(show);
      }
    });
  }
});
