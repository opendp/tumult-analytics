{% if not obj.display %}
:orphan:
{% endif %}

{% set name_parts = obj.name.split('.') %}

{% if obj.name == package_name %}
   {% set module_title = "API Reference" %}
{% elif name_parts|length > 2 %}
   {% set module_title = obj.short_name %}
{% else %}
   {% set module_title = obj.name %}
{% endif %}

*******{{ "*" * module_title|length }}
{{ module_title }}
*******{{ "*" * module_title|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|prepare_docstring|indent(3) }}
{% endif %}

{% block subpackages_and_modules %}
{% set visible_subpackages_modules = obj.subpackages|selectattr("display")|list + obj.submodules|selectattr("display")|list%}
{% if visible_subpackages_modules %}
Modules
=======

.. list-table::

{% for subpackage in visible_subpackages_modules|sort %}
   * - :mod:`.{{ subpackage.short_name }}`
     - {{ subpackage.summary }}
{% endfor %}

.. toctree::
   :hidden:

{% for subpackage in visible_subpackages_modules|sort %}
   {{ subpackage.short_name }}/index.rst
{% endfor %}

{% endif %}
{% endblock %}


{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}

{% if visible_children %}

{% set visible_data = visible_children|selectattr("type", "equalto", "data")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}



{% if visible_data %}
Data
====

{% if visible_data|length >= 5 %}
.. list-table::

   {% for obj_item in visible_data %}
   * - :data:`.{{ obj_item.short_name }}`
     - {{ obj_item.summary }}
   {% endfor %}
{% endif %}

{% for obj_item in visible_data %}
{{ obj_item.rendered|indent(0) }}
{% endfor %}
{% endif %}

{% if visible_functions %}

Functions
=========

{% if visible_functions|length >= 1 %}
.. list-table::

   {% for obj_item in visible_functions %}
   * - :func:`.{{ obj_item.short_name }}`
     - {{ obj_item.summary }}
   {% endfor %}
{% endif %}

{% for obj_item in visible_functions %}
{{ obj_item.rendered|indent(0) }}

{% endfor %}
{% endif %}

{% if visible_classes %}
Classes
=======

{% if visible_classes|length >= 1 %}
.. list-table::

   {% for obj_item in visible_classes %}
   * - :class:`.{{ obj_item.short_name }}`
     - {{ obj_item.summary }}
   {% endfor %}
{% endif %}

{% for obj_item in visible_classes %}
   {{ obj_item.rendered|indent(0) }}
{% endfor %}
{% endif %}

{% if visible_exceptions %}
Exceptions
==========

{% if visible_exceptions|length >= 3 %}
.. list-table::

   {% for obj_item in visible_exceptions %}
   * - :exc:`.{{ obj_item.short_name }}`
     - {{ obj_item.summary }}
   {% endfor %}
{% endif %}

{% for obj_item in visible_exceptions %}
   {{ obj_item.rendered|indent(0) }}
{% endfor %}
{% endif %}

{% endif %}

{% endblock %}
