# usecase_kartika/templatetags/kartika_extras.py
from django import template

register = template.Library()

@register.filter(name='get_field_label')
def get_field_label(form, field_name_string):
    """
    Mengembalikan label dari sebuah field form berdasarkan nama field tersebut.
    Jika label tidak ditemukan, kembalikan nama field yang diformat.
    """
    try:
        if hasattr(form, 'fields') and field_name_string in form.fields:
            return form.fields[field_name_string].label
    except KeyError:
        pass # Jika field tidak ditemukan di form.fields (seharusnya tidak terjadi jika key dari input_details)
    
    # Fallback jika label tidak ada atau field tidak ditemukan (sebagai pengaman)
    return field_name_string.replace('_', ' ').capitalize()