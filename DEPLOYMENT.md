# Deployment

## Vercel

Import the GitHub repository into Vercel and keep the project root at the repository root. Vercel uses `vercel.json` to route requests to the Flask application.

Add these environment variables in the Vercel project settings for Production, Preview, and Development:

- `GOOGLE_API_KEY`: Google AI Studio API key used by the Gemini client.
- `SECRET_KEY`: A stable, randomly generated secret used to sign login sessions.

Redeploy after adding or changing environment variables. The generated vector index at `data/index.json` is included in the repository and is required for chat retrieval.

The current SQLite database is placed under `/tmp` on Vercel so the function can write to it. Vercel's filesystem is temporary and each function instance may have its own database, so accounts and chat history are not durable or shared across instances. Use a managed database and update the database layer before relying on this deployment for production user data.