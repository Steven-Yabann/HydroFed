# HydroFed dashboard

React 19 + Vite. No UI or chart libraries: charts are hand-built SVG in `src/components/charts.jsx`.

```bash
npm install
npm run dev     # http://localhost:5173, talks to VITE_API_BASE_URL
npm run lint
npm run build
```

- `src/lib/`: API client with the session token, auth context, hash router, formatting
- `src/components/`: icons, shared UI pieces, charts
- `src/pages/`: fleet overview, plant detail, alerts, model performance, sign-in
- `src/pages/admin/`: federated training, plant nodes, users and access

Navigation follows the signed-in user's permissions from `/api/auth/me`, so adding a role in `hydrofed.auth.ROLE_PERMISSIONS` needs no frontend change unless the role gets new pages.

Design notes: white surfaces with layered greens for structure. Colour signals abnormal state (watch, alarm, alert severity), and every status also carries an icon and a word. Chart series colours are checked for colour-blind separation.
