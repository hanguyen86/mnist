"""
Handwritten Digit Recognition
Licence: BSD
Author : Hoang Anh Nguyen
"""

import os
from app import app

#----------------------------------------
# launch
#----------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    app.run(host='0.0.0.0', port=port)