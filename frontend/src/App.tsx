import { RouterProvider } from "react-router-dom";
import { router } from "./router";

/**
 * Top-level App — wraps the router in a fragment so the QueryClient
 * + BrowserRouter from main.tsx stay above it. The router owns the
 * page tree from here down (AppShell + nested routes).
 */
export default function App(): JSX.Element {
  return <RouterProvider router={router} />;
}
