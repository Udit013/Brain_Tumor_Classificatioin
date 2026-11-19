/** Shared by the root layout (pre-paint script) and the client-side ThemeToggle. */
export const THEME_KEY = "nc-theme";

/** Runs before first paint: applies a stored explicit light/dark choice. */
export const THEME_SCRIPT = `try{var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;
