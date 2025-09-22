FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps

# Build the production assets
COPY . .
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app

# Lightweight static server that respects $PORT
RUN npm install -g serve

# Copy built assets from builder
COPY --from=builder /app/dist /app/dist

# Railway provides a PORT env var; fall back to 8080
EXPOSE 8080

CMD ["sh", "-c", "serve -s /app/dist -l ${PORT:-8080}"]
