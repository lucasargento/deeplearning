class AppCfg:
    ACCESS_TOKEN = 'APP_USR-8701980465483027-111623-81a27f92675771e29d6fa4673f47334b-158688410'
    
    
    
# https://midomain.com.ar/auth/mercadolibre?code=TG-61943f19e7d893001b881cc4-158688410&state=


# https://auth.mercadolibre.com.ar/authorization?response_type=code&client_id=8701980465483027&redirect_uri=https://midomain.com.ar/auth/mercadolibre

# curl -X POST \
# -H 'accept: application/json' \
# -H 'content-type: application/x-www-form-urlencoded' \
# 'https://api.mercadolibre.com/oauth/token' \
# -d 'grant_type=authorization_code' \
# -d 'client_id=8701980465483027' \
# -d 'client_secret=EfpklikNAH6KwJ2FO46y8bJQ5Dyb0e5s' \
# -d 'code=TG-619440b8967f8b001a44d5a9-158688410' \
# -d 'redirect_uri=https://midomain.com.ar/auth/mercadolibre'

# curl -X GET -H 'Authorization: Bearer APP_USR-8701980465483027-111623-81a27f92675771e29d6fa4673f47334b-158688410' https://api.mercadolibre.com/sites/MLA/search?category=MLA1055
