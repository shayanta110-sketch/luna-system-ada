# Nexus Router

A lightweight, flexible routing solution for modern web applications.

## Features

- 🚀 **Zero dependencies** – small bundle size
- ⚡ **Simple API** – intuitive route definitions
- 🔀 **Dynamic segments** – `/users/:id` style parameters
- 🌟 **Query parsing** – automatic query string handling
- 🧩 **Middleware support** – pre and post route hooks
- 📦 **Framework agnostic** – works with React, Vue, vanilla JS, and more
- 🛣️ **Nested routes** – organize complex routing hierarchies
- 🔍 **Path matching** – regex and custom matchers

## Installation

```bash
npm install nexus-router
```

or

```bash
yarn add nexus-router
```

## Basic Usage

```javascript
import { Router } from 'nexus-router';

// Create a router instance
const router = new Router();

// Define routes
router.get('/', () => {
  console.log('Home page');
});

router.get('/users/:id', (params, query) => {
  console.log(`User ID: ${params.id}`);
  console.log(`Query params:`, query);
});

router.post('/api/data', (params, query, body) => {
  console.log('Received data:', body);
});

// Navigate
router.resolve('/users/42?tab=profile');
// Output: User ID: 42
//         Query params: { tab: 'profile' }
```

## API Reference

### `router.get(path, handler)`
Register a GET route handler.

### `router.post(path, handler)`
Register a POST route handler.

### `router.put(path, handler)`
Register a PUT route handler.

### `router.delete(path, handler)`
Register a DELETE route handler.

### `router.resolve(path, method = 'GET', body = null)`
Match and execute the appropriate route handler.

### `router.use(middleware)`
Add global middleware functions.

## Advanced Example

```javascript
// Middleware for authentication
router.use((ctx, next) => {
  if (!ctx.isAuthenticated) {
    console.log('Unauthorized');
    return;
  }
  next();
});

// Nested routes
const apiRouter = new Router();
apiRouter.get('/posts', () => console.log('List posts'));
apiRouter.get('/posts/:id', () => console.log('Get post'));

router.mount('/api', apiRouter);

router.resolve('/api/posts/5'); // Output: Get post
```

## License

MIT