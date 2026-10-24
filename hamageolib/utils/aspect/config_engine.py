"""
config_engine.py

Core configuration and rule engine for mapping high-level scientific
parameters onto ASPECT prm_dict and WorldBuilder wb_dict structures.

This module provides:
- AspectModelConfig: generic validated configuration base class
- Rule: base class for transformation rules
- RuleEngine: applies rules with dependency validation

Users are expected to:
- Subclass AspectModelConfig and define their own parameter schema
- Import Rule and define custom Rule subclasses
- Register rules with RuleEngine
"""

from __future__ import annotations


class Rule:
    """
    Base class for all transformation rules.

    A Rule encapsulates one logical transformation that maps high-level
    model parameters (from the config) onto low-level structures
    such as prm_dict and wb_dict.

    Subclasses are expected to define three optional class attributes:

    - requires: list[str]
        Names of parameters that must be available in the config.
        These are validated by the RuleEngine before apply() is called.
        Example:
            requires = ["slab_age", "dip_angle"]

    - defaults: dict[str, object] = {}
        Parameters that are with available default values
        These are added by the RuleEngine before apply() is called.
        Example:
            defaults = {"slab_age": 80e6, "dip_angle": np.pi/3.0}

    - requires_comments: dict[str, str]
      Optional human-readable documentation for each required parameter.
      This is metadata only and does not affect validation or execution.

      Example:
          requires = ["slab_age", "dip_angle"]
          requires_comments = {
              "slab_age": "Age of the incoming plate in years",
              "dip_angle": "Initial slab dip in radians"
          }

    - provides: list[str]
        Names of keys that this rule writes into the shared `context` dict.
        This is optional metadata, mainly useful for:
          - documenting what this rule produces
          - enabling future dependency tracking between rules
          - avoiding accidental key collisions in large workflows

        Example:
            provides = ["slab_length_km", "trench_position"]

    Subclasses must implement:
        apply(config, prm_dict, wb_dict, context)
    """

    # Parameters required from the dict
    requires: list[str] = []

    # Parameters' default values from the dict
    defaults: dict[str, object] = {}

    # Optional comments for required parameters (metadata only)
    requires_comments: dict[str, str] = {}

    # Keys written into the shared context dictionary (optional metadata)
    provides: list[str] = []

    def add_default(self, config: dict):
        """
        Add default values for missing required configuration entries.

        This function examines the provided configuration dictionary, determines which
        required parameters are missing, and uses the default values defined in the
        class to fill those missing entries when available.

        Parameters:
            config (dict): A dictionary containing configuration key–value pairs,
                        typically parsed from a configuration file.

        Returns:
            dict: The updated configuration dictionary with defaults applied where needed.
        """
        # Collect names of parameters that have non-None default values defined in the class
        defaulted = {
            name for name, value in self.defaults.items()
            if value is not None
        }

        # Convert the list (or iterable) of required parameters into a set
        required = set(self.requires)

        # Identify which required parameters are missing from the available configuration
        missing = required - {
            name for name, value in config.items()
            if value is not None
        }

        # Fill missing required parameters using defaults when available
        for name in missing:
            if name in defaulted:
                config[name] = self.defaults[name]

        # Recompute missing after applying defaults
        still_missing = required - {
            name for name, value in config.items()
            if value is not None
        }

        # Raise error if some required parameters are still missing
        if still_missing:
            raise ValueError(
                f"Missing required configuration parameters without defaults: {sorted(still_missing)}"
            )

        return config

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply this rule.

        Args:
            config: AspectModelConfig instance (validated by RuleEngine)
            prm_dict: dict representation of the ASPECT .prm file
            wb_dict: dict representation of the WorldBuilder .wb file
            context: shared dict for passing derived quantities between rules
        """
        raise NotImplementedError("Rule subclasses must implement apply()")


class RuleConflictError(Exception):
    """
    Raised when two or more rules produce incompatible or conflicting effects.
    """
    pass

class RuleInternalError(Exception):
    """
    Raised when the options in one rule conflict each other.
    """
    pass


class RuleEngine:
    """
    Applies a list of rules to prm_dict and wb_dict with validation.
    """

    def __init__(self, rules: list[Rule]):
        self.rules = rules

    def add_default(self, config: dict):
        """
        Ensure all required parameters for all rules are available.
        Args:
            config: AspectModelConfig instance (validated by RuleEngine)
        """
        for rule in self.rules:
            rule.add_default(config)

    def apply_all(self, config, prm_dict, wb_dict, *, doc_keys=None):
        """
        Apply all rules in order.

        Returns:
            tuple:
                context (dict): Dictionary containing all derived values.
                documentation (list[dict]): Structured documentation of rule inputs.
        """
        # First, ensure all required parameters for all rules are available.
        self.add_default(config)

        # Fix keys to documents
        all_keys = [key for key, _ in config.items()]
        if doc_keys is None:
            doc_keys = all_keys

        context = {}
        full_documentation = []
        documentation = []

        for rule in self.rules:

            # apply the rule
            rule.apply(config, prm_dict, wb_dict, context)

            # document the parameters
            full_rule_doc = {
                "rule": rule.__class__.__name__,
                "requires": [],
                "provides": []
            }

            rule_doc = {
                "rule": rule.__class__.__name__,
                "requires": [],
                "provides": []
            }

            for name in rule.requires:
                value = config.get(name)
                comment = getattr(rule, "requires_comments", {}).get(name, "")

                full_rule_doc["requires"].append({
                    "name": name,
                    "value": value,
                    "comment": comment
                })

                if name in doc_keys:
                    rule_doc["requires"].append({
                        "name": name,
                        "value": value,
                        "comment": comment
                    })

            for name in rule.provides:
                value = context.get(name)
                comment = getattr(rule, "provides_comments", {}).get(name, "")

                full_rule_doc["provides"].append({
                    "name": name,
                    "value": value,
                    "comment": comment
                })

                if name in doc_keys:
                    rule_doc["provides"].append({
                        "name": name,
                        "value": value,
                        "comment": comment
                    })

            full_documentation.append(full_rule_doc)
            documentation.append(rule_doc)

        return context, full_documentation, documentation

    def get_required_variables(self, config) -> dict:
        """
        Return all required variable names and their values.

        This collects the union of all variables listed in `requires`
        across all rules and returns their values from the config.

        Parameters:
            config: AspectModelConfig instance (or dict-like object)

        Returns:
            dict: Mapping from variable name to value.
        """
        # Ensure defaults are applied before reading values
        self.add_default(config)

        variables = {}

        for rule in self.rules:
            for name in rule.requires:
                variables[name] = config.get(name)

        return variables

    def render_docs_markdown(self, documentation: list[dict]) -> str:
        """
        Render rule documentation into a Markdown string.

        Parameters:
            documentation (list[dict]): Documentation structure returned by apply_all().

        Returns:
            str: Markdown-formatted documentation.
        """
        lines = []
        lines.append("# Rule Engine Documentation\n")

        for rule_doc in documentation:
            rule_name = rule_doc.get("rule", "UnnamedRule")
            lines.append(f"## {rule_name}\n")

            requires = rule_doc.get("requires", [])

            if not requires:
                lines.append("_No required parameters._\n")
                continue

            lines.append("| Parameter (required) | Value | Comment |")
            lines.append("|-----------|-------|---------|")

            for entry in requires:
                name = entry.get("name", "")
                value = entry.get("value", "")
                comment = entry.get("comment", "")

                lines.append(f"| `{name}` | `{value}` | {comment} |")

            lines.append("")  # blank line between rules

            provides = rule_doc.get("provides", [])

            if not provides:
                lines.append("_No provided parameters._\n")
                continue

            lines.append("| Parameter (provided) | Value | Comment |")
            lines.append("|-----------|-------|---------|")

            for entry in provides:
                name = entry.get("name", "")
                value = entry.get("value", "")
                comment = entry.get("comment", "")

                lines.append(f"| `{name}` | `{value}` | {comment} |")

            lines.append("")  # blank line between rules

        return "\n".join(lines)


    def render_docs_table(self, documentation: list[dict]) -> str:
        """
        Render rule documentation into a plain-text table.

        Parameters:
            documentation (list[dict]): Documentation structure returned by apply_all().

        Returns:
            str: Plain-text formatted table.
        """
        lines = []

        for rule_doc in documentation:
            rule_name = rule_doc.get("rule", "UnnamedRule")
            requires = rule_doc.get("requires", [])
            provides = rule_doc.get("provides", [])

            parameters = requires + provides

            lines.append(f"Rule: {rule_name}")
            lines.append("-" * (6 + len(rule_name)))

            if not parameters:
                lines.append("  (No required parameters)")
                lines.append("")
                continue

            # Determine column widths
            name_width = max(len("Parameter"), *(len(r["name"]) for r in parameters))
            value_width = max(len("Value"), *(len(str(r["value"])) for r in parameters))
            comment_width = max(len("Comment"), *(len(r["comment"]) for r in parameters))

            # Header
            lines.append(
                f"{'Parameter'.ljust(name_width)} | "
                f"{'Value'.ljust(value_width)} | "
                f"{'Comment'.ljust(comment_width)}"
            )
            lines.append(
                f"{'-' * name_width}-+-"
                f"{'-' * value_width}-+-"
                f"{'-' * comment_width}"
            )

            # Rows
            for entry in parameters:
                lines.append(
                    f"{entry['name'].ljust(name_width)} | "
                    f"{str(entry['value']).ljust(value_width)} | "
                    f"{entry['comment'].ljust(comment_width)}"
                )

            lines.append("")  # blank line between rules

        return "\n".join(lines)
