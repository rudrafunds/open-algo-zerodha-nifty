# rf_algo_main.py
import json
import sys
import os
from datetime import datetime, timedelta
import subprocess
from kiteconnect import KiteConnect

class UnifiedConfigManager:
    """Handles all configuration management and Kite Connect operations"""
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = None
        self.kite = None
        self.python_cmd = self._get_python_command()
        self.load_config()

    def _get_python_command(self):
        """Determine system-appropriate Python command"""
        if sys.executable:
            return [sys.executable]
        for cmd in ['python3', 'python', 'py']:
            try:
                subprocess.check_call([cmd, '--version'], 
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
                return [cmd]
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        raise EnvironmentError("No valid Python interpreter found")

    def load_config(self):
        """Load and validate configuration"""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in config file: {self.config_path}")

        # Validate required configurations
        if 'kite' not in self.config:
            raise ValueError("Missing 'kite' section in config")
        
        required_kite = ['api_key', 'api_secret', 'refresh_token']
        for key in required_kite:
            if key not in self.config['kite'] or not self.config['kite'][key]:
                raise ValueError(f"Missing or empty kite.{key} in config")

    def save_config(self):
        """Save updated configuration back to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get_token_status(self):
        """Determine token status based on last update time"""
        last_update_str = self.config['kite'].get('last_update_time')
        if not last_update_str:
            return "expired"
            
        last_update = datetime.strptime(last_update_str, "%d-%m-%Y %H:%M:%S")
        today = datetime.today().date()
        
        if last_update.date() < today:
            return "needs_refresh"
        elif last_update.date() == today:
            return "valid"
        return "expired"

    def update_refresh_token(self):
        """Update refresh token from user input"""
        new_token = input("Enter refresh token: ").strip()
        self.config['kite']['refresh_token'] = new_token
        self.save_config()

    def reset_persistence(self):
        """Reset persistence data in config"""
        self.config['persistent']['report']['opening_funds'] = 0
        self.config['persistent']['report']['today_pnl'] = 0
        self.config['persistent']['report']['today_change'] = 0
        self.config['persistent']['report']['positions']['total'] = 0
        self.config['persistent']['report']['positions']['win_percent'] = 0
        self.config['persistent']['report']['positions']['target'] = 0
        self.config['persistent']['report']['positions']['break_even'] = 0
        self.config['persistent']['report']['positions']['stop_loss'] = 0
        self.config['persistent']['report']['positions']['CE'] = 0
        self.config['persistent']['report']['positions']['PE'] = 0
        self.config['persistent']['report']['last_update_time'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print("Reset persistent reports")

    def generate_access_token(self):
        """Generate and persist new access token using refresh token"""
        try:
            self.kite = KiteConnect(api_key=self.config['kite']['api_key'])
            session = self.kite.generate_session(
                self.config['kite']['refresh_token'],
                api_secret=self.config['kite']['api_secret']
            )
            self.config['kite']['access_token'] = session['access_token']
            self.config['kite']['last_update_time'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            self.reset_persistence()
            self.save_config()
            return session['access_token']
        except Exception as e:
            raise RuntimeError(f"Token generation failed: {str(e)}")

    def run_algo_script(self):
        """Execute the appropriate algorithm script"""
        message = "Starting "
        if self.config['system']['selling']:
            script = 'src/rf_algo_v9_sell.py'
            message += "Selling "
        else:
            script = 'src/rf_algo_v10.py'
            message += "Buying "
        message += "Algorithm..."
        
        print(message)
        subprocess.run(self.python_cmd + [script])

    def handle_workflow(self):
        """Main execution flow controller"""
        print("RF Algo initializing...")
        status = self.get_token_status()

        if status == "needs_refresh":
            print("Kite: Access token needs refresh")
            self.update_refresh_token()
            print("Generating new access token...")
            self.generate_access_token()
            self.run_algo_script()
        elif status == "valid":
            print("Kite: Access token is valid")
            self.run_algo_script()
        else:
            print("Token status: Expired")
            print(f"Last update: {self.config['kite'].get('last_update_time', 'never')}")
            print("Please manually verify the configuration.")

if __name__ == "__main__":
    try:
        config_manager = UnifiedConfigManager()
        config_manager.handle_workflow()
    except Exception as e:
        print(f"Critical error occurred: {str(e)}")
        sys.exit(1)