import os
import random
import string
from pathlib import Path

def generate_bloat(size_kb=5):
    """Generates a block of useless but realistic-looking bloat (comments, JSON, SVGs)."""
    bloat = "\n/**\n * Auto-generated Boilerplate Documentation Section\n"
    for _ in range(size_kb * 10):
        line = ''.join(random.choices(string.ascii_letters + " ", k=80))
        bloat += f" * {line}\n"
    bloat += " */\n"
    
    # Add a massive dummy JSON object
    bloat += "const DUMMY_ASSETS = {\n"
    for i in range(10):
        bloat += f"  asset_{i}: 'data:image/svg+xml;base64,{'x' * 500}',\n"
    bloat += "};\n"
    
    return bloat

def create_vue_maze(base_dir="target_repo", num_components=100, num_stores=10):
    """Generates a brutal, enterprise-bloated Vue/TS repository for benchmarking."""
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Create Directories
    dirs = ['src/components', 'src/views', 'src/stores', 'src/utils', 'src/services']
    for d in dirs:
        os.makedirs(Path(base_dir) / d, exist_ok=True)

    # 2. Build the "Deep Dependency Chain" (The SRM Target Path)
    # Logger -> HttpUtils -> ApiClient -> AuthStore -> HeaderWidget
    
    logger_content = f"""export const logger = {{ log: (msg: string) => console.log('[Log]', msg) }};{generate_bloat(10)}"""
    with open(Path(base_dir) / 'src/utils/logger.ts', 'w') as f:
        f.write(logger_content)

    http_utils = f"""import {{ logger }} from './logger';
export const clearSessionCookies = () => {{ logger.log('clearing session cookies'); }};
export const post = (url: string, data: any) => {{ 
    logger.log('POST ' + url);
    if (url === '/logout') clearSessionCookies();
    return Promise.resolve({{ data }});
}};{generate_bloat(10)}"""
    with open(Path(base_dir) / 'src/utils/http_utils.ts', 'w') as f:
        f.write(http_utils)

    api_client = f"""import {{ post }} from '../utils/http_utils';
export const api = {{ 
    login: (u: string, p: string) => post('/login', {{ u, p }}),
    logout: () => post('/logout', {{}})
}};{generate_bloat(10)}"""
    with open(Path(base_dir) / 'src/services/api_client.ts', 'w') as f:
        f.write(api_client)

    auth_store = f"""import {{ defineStore }} from 'pinia';
import {{ api }} from '../services/api_client';
export const useAuthStore = defineStore('auth', {{
  state: () => ({{ user: {{ id: 1, role: 'admin' }}, token: 'jwt_xyz' }}),
  actions: {{ 
    async login(u: string, p: string) {{ await api.login(u, p); }},
    async logout() {{ await api.logout(); this.token = null; }} 
  }}
}});{generate_bloat(10)}"""
    with open(Path(base_dir) / 'src/stores/authStore.ts', 'w') as f:
        f.write(auth_store)

    header_widget = f"""<script setup lang="ts">
import {{ useAuthStore }} from '../stores/authStore';
const auth = useAuthStore();
</script>
<template><div>{{ auth.user.role }}</div></template>
<!-- {generate_bloat(20)} -->"""
    with open(Path(base_dir) / 'src/components/HeaderWidget.vue', 'w') as f:
        f.write(header_widget)

    user_settings = f"""<script setup lang="ts">
import {{ useAuthStore }} from '../stores/authStore';
const auth = useAuthStore();
const handleLogout = () => {{ auth.logout(); }};
</script>
<template>
  <div class="settings-view">
    <h1>User Settings</h1>
    <button @click="handleLogout">Logout</button>
  </div>
</template>
<!-- {generate_bloat(20)} -->"""
    with open(Path(base_dir) / 'src/views/UserSettings.vue', 'w') as f:
        f.write(user_settings)

    # 3. Create "Red Herrings" (The RAG Killers)
    # These look relevant to 'auth' or 'user' but are logically disconnected
    store_names = ['billingStore', 'userPreferenceStore', 'metricsStore', 'notificationStore', 'permissionStore']
    for name in store_names:
        herring_content = f"""import {{ defineStore }} from 'pinia';
// This store manages {name} logic and user state related to permissions
export const use{name[0].upper()}{name[1:]} = defineStore('{name}', {{
  state: () => ({{ data: [], loading: false, user_id: 1 }}),
  actions: {{ fetchData() {{ this.loading = true; }} }}
}});{generate_bloat(30)}"""
        with open(Path(base_dir) / f'src/stores/{name}.ts', 'w') as f:
            f.write(herring_content)

    mock_logout = f"""// This file is a mock implementation for testing
// user settings, logout flow, and clear session logic.
// However, it is never actually imported by the real application!
export const handleUserLogoutSettings = () => {{
    console.log("Mocking the user settings logout event");
}};
export const clearSession = () => {{
    console.log("Mock clear session handler");
}};
{generate_bloat(40)}"""
    with open(Path(base_dir) / 'src/utils/mockLogoutHandler.ts', 'w') as f:
        f.write(mock_logout)

    # 4. Massively Inflate Noise (The Token Burners)
    for i in range(num_components):
        comp_content = f"""<script setup lang="ts">
import {{ ref }} from 'vue';
const count = ref({i});
</script>
<template><div>Component {i} Boilerplate</div></template>
<!-- {generate_bloat(random.randint(20, 50))} -->"""
        with open(Path(base_dir) / f'src/components/Widget{i}.vue', 'w') as f:
            f.write(comp_content)

    print(f"Successfully generated BRUTAL enterprise maze.")
    print(f"- 100+ bloated components")
    print(f"- 5-level deep dependency chain")
    print(f"- 5 high-similarity Red Herring stores")

if __name__ == "__main__":
    create_vue_maze()
