import os

# Bind to the port specified by the environment
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker configuration
workers = 3
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
errorlog = "-"
loglevel = "info"
accesslog = "-"

# Set the process name
proc_name = "property_valuation_app"

# Preload application
preload_app = True