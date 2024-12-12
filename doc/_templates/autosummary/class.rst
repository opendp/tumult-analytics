{{ objname | escape | underline}}

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2024

.. testcode::
{% if "." in objname %}
   from {{ module }} import {{ objname.split(".")[0] }}
{% else %}
   from {{ module }} import {{ objname }}
{% endif %}

.. currentmodule:: {{ module }}

{# Builders are nested classes, so need to be specified with the fully qualified name
 # due to a decade-old limitation of autodoc. They are also implemented by inheriting
 # from Mixin classes, so we need to document inherited methods. This implementation
 # detail leads to arbitrary-looking ordering, so we use alphabetical order instead. #} 
{% if objname.endswith(".Builder") %} 

.. autoclass:: {{ module }}::{{ objname }}
   :members:
   :no-show-inheritance:
   :inherited-members:
   :member-order: alphabetical

{# For some classes, we only document a subset of members in the main documentation
 # page. We then list attributes and methods at the bottom of the page. #} 
{% elif objname in showed_members %} 

.. autoclass:: {{ objname }}
   {% if showed_members[objname] %}:members: {{ ",".join(showed_members[objname]) }}{% endif %}

{% if attributes %}
.. rubric:: {{ _('Attributes') }}
.. autosummary::
   :nosignatures:

   {% for attr in attributes %}
   {{ objname }}.{{ attr }}
   {%- endfor %}
{% endif %}
{% if methods %}
.. rubric:: {{ _('Methods') }}
.. autosummary::
   :nosignatures:

   {% for method in methods %}
   {% if method != "__init__" and method not in showed_members[objname] %}
   {{ objname }}.{{ method }}
   {% endif %}
   {%- endfor %}
{% endif %}

{# Otherwise, we document a class contents normally. #} 
{% else %} 

.. autoclass:: {{ objname }}
   {% if objname in show_init -%}
   :special-members: __init__
   {% endif -%}
   :members:

{% endif %} 
