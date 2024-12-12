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

.. auto{{ objtype }}:: {{ objname }}
