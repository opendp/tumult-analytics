{{ objname | escape | underline}}

..
    SPDX-License-Identifier: CC-BY-SA-4.0
    Copyright Tumult Labs 2024

.. testcode::

   from {{ module }} import {{ class }}

.. currentmodule:: {{ module }}

.. automethod:: {{ objname }}

{# Adding companion methods if needed #}
{% if objname in companion_methods %} 

.. testcode::

   from {{ module }} import {{ companion_methods[objname].split(".")[0] }}

.. automethod:: {{ companion_methods[objname] }}
{% endif %} 

{# Adding companion classes if needed #}
{% if objname in companion_classes %} 

.. testcode::

   from {{ module }} import {{ companion_classes[objname] }}

.. autoclass:: {{ companion_classes[objname] }}
   :members:

{% endif %} 
