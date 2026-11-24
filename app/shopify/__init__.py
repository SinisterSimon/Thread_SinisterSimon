"""Read-only connection to a real Shopify store (v2)."""
from app.shopify.client import ShopifyClient, ShopifyError, client_from_env
from app.shopify.sync import SyncResult, sync_store

__all__ = ["ShopifyClient", "ShopifyError", "SyncResult", "client_from_env", "sync_store"]
