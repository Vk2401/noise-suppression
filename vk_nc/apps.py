from django.apps import AppConfig


class VkNcConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'vk_nc'
    
    def ready(self):
        from django.template.defaultfilters import register
        from .custom_filters import basename

        # Register your custom filter
        register.filter('basename', basename)