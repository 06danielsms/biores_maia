# Frontend BIORES Maia

Aplicación SPA construida con React 19, Vite y Ant Design. Proporciona vistas para monitorear KPIs, explorar el corpus, gestionar traducciones, orquestar resúmenes y supervisar operaciones.

## Scripts disponibles

```bash
npm run dev       # Servidor de desarrollo (http://localhost:5173)
npm run build     # Build de producción
npm run preview   # Previsualización de la build
npm run lint      # Linter con ESLint flat config
npm run test      # Tests con Vitest + Testing Library
```

## Configuración de entorno

1. Copia `.env.local.example` a `.env.local`.
2. Ajusta `VITE_API_URL` según tu entorno:
   - Desarrollo local: `http://localhost:8000`
   - Docker/reverse proxy: la URL pública que exponga el backend (por ejemplo `https://tu-dominio/api`).
3. Reinicia `npm run dev` cada vez que cambies variables de entorno.

La aplicación usa `retry: 0` y `refetchOnWindowFocus: false` en React Query para evitar tormentas de reintentos cuando no hay backend disponible. Un banner en la parte superior permite reintentar manualmente cuando la API está caída.

### Scripts de ingestión (backend)

Antes de servir el frontend con datos reales, carga la información en MongoDB:

```bash
pip install -r backend/requirements.txt
./scripts/seed_local_dev.sh
```

Cada script acepta `--mongo`, `--database` y `--drop` para apuntar a instancias personalizadas.

## Estructura

```
src/
├── components/       # Widgets reutilizables (navegación, KPIs, timeline, etc.)
├── config/           # Configuración central (menús, rutas)
├── hooks/            # Hooks de datos basados en React Query
├── layouts/          # Layout principal de la aplicación
├── pages/            # Vistas alineadas a las fases del roadmap
├── services/         # Cliente HTTP y utilidades de red
└── store/            # Estado global (Zustand)
```

## Tests

Los tests viven en `src/**/__tests__` y se ejecutan con:

```bash
npm run test -- --run
```

Vitest está configurado con `jsdom` y `@testing-library` a través de `src/setupTests.js`.

## Mock API con MSW

Puedes activar mocks realistas de los endpoints del backend encendiendo `VITE_ENABLE_MOCKS=true` en tu `.env.local`.

- Usa los mismos contratos que el backend (`/health/`, `/datasets/state`, `/jobs/status`, etc.).
- Ideal para desarrollo sin backend o para demos offline.
- Se apaga dejando `VITE_ENABLE_MOCKS=false` o eliminando la variable.

Los tests también utilizan los handlers de MSW para garantizar que no se hagan llamadas reales a red.
