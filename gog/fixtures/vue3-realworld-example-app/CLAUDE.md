# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vue 3 implementation of the [RealWorld](https://github.com/gothinkster/realworld) example app ("Conduit" — a Medium clone). Uses Composition API with `<script setup>`, TypeScript, Pinia, and Vue Router with hash history.

## Commands

```bash
pnpm install              # Install dependencies (requires pnpm, Node >= 20)
pnpm dev                  # Dev server on port 4173
pnpm build                # Production build
pnpm lint                 # ESLint with auto-fix
pnpm type-check           # vue-tsc type checking

# Unit tests (Vitest + Testing Library + happy-dom)
pnpm test:unit            # Run all unit tests
npx vitest run src/components/AppLink.spec.ts   # Run a single test file
npx vitest --watch        # Watch mode

# E2E tests (Playwright, builds first then serves)
pnpm test:e2e             # Run all E2E tests (builds app first)
pnpm test:e2e:ui          # Playwright UI mode
npx playwright test playwright/specs/home.spec.ts  # Single E2E test file

# API client regeneration from OpenAPI spec
pnpm generate:api         # Downloads spec and generates src/services/api.ts
```

## Architecture

### Routing & Pages
- `src/router.ts` — All routes defined with lazy-loaded page components. Exports typed `routerPush()` helper and `AppRouteNames` type for type-safe navigation.
- `src/pages/` — Page-level components (Home, Article, EditArticle, Login, Register, Profile, Settings). Home.vue is shared across global-feed, my-feed, and tag routes.

### State Management
- `src/store/user.ts` — Single Pinia store for user auth state. Auth token persisted to localStorage via `Storage` utility. `isAuthorized()` function export used for route guards.

### API Layer
- `src/services/api.ts` — **Auto-generated** from OpenAPI spec via `swagger-typescript-api`. Do not edit manually.
- `src/services/index.ts` — API client instance configured with base URL from `VITE_API_HOST` env var and token-based auth via `securityWorker`.
- `src/config.ts` — Reads `VITE_API_HOST` from environment.

### Composables
- `src/composable/` — Vue composables for domain logic (articles list/pagination, favorites, follows, profile, tags). `use-articles.ts` derives article feed type from current route name.

### Plugins
- `src/plugins/global-components.ts` — Registers global components
- `src/plugins/marked.ts` — Markdown rendering (marked + insane for sanitization)
- `src/plugins/set-authorization-token.ts` — Restores auth token on app init

### Path Alias
- `src` is aliased to the `src/` directory (configured in vite.config.ts). Use `import { x } from 'src/...'` style imports.

## Testing

### Unit Tests
- Co-located with source: `src/**/*.spec.ts`
- Uses MSW for API mocking via `setupMockServer()` helper in `src/utils/test/test.utils.ts`
- `renderOptions()` helper provides router, Pinia store, and global components for Testing Library renders
- `asyncWrapper()` wraps async components in `<Suspense>` for testing
- Test fixtures in `src/utils/test/fixtures.ts`

### E2E Tests
- Located in `playwright/specs/`
- Uses page object pattern: `playwright/page-objects/`
- Runs against built app on port 4173

## Code Style
- ESLint config: `@mutoe/eslint-config` (based on Anthony Fu's config)
- Vue custom events must use kebab-case
- `src/services/api.ts` is excluded from linting (auto-generated)
- Pre-commit hook runs `lint-staged` (ESLint on all staged files)
- Pre-push hook runs type-check and build
