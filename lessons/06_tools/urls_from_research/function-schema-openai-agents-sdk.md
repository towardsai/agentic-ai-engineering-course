# `Function schema`

### FuncSchema`dataclass`

Captures the schema for a python function, in preparation for sending it to an LLM as a tool.


```python
@dataclass
class FuncSchema:
    """
    Captures the schema for a python function, in preparation for sending it to an LLM as a tool.
    """
    name: str
    """The name of the function."""
    description: str | None
    """The description of the function."""
    params_pydantic_model: type[BaseModel]
    """A Pydantic model that represents the function's parameters."""
    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters, derived from the Pydantic model."""
    signature: inspect.Signature
    """The signature of the function."""
    takes_context: bool = False
    """Whether the function takes a RunContextWrapper argument (must be the first argument)."""
    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input."""
    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """
        Converts validated data from the Pydantic model into (args, kwargs), suitable for calling
        the original function.
        """
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False
        # Use enumerate() so we can skip the first parameter if it's context.
        for idx, (name, param) in enumerate(self.signature.parameters.items()):
            # If the function takes a RunContextWrapper and this is the first parameter, skip it.
            if self.takes_context and idx == 0:
                continue
            value = getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args and mark that *args is now seen
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                # e.g. **kwargs handling
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, add to keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # For KEYWORD_ONLY parameters, always use keyword args.
                keyword_args[name] = value
        return positional_args, keyword_args
```

#### name`instance-attribute`

```python
name: str
```
The name of the function.

#### description`instance-attribute`

```python
description: str | None
```
The description of the function.

#### params_pydantic_model`instance-attribute`

```python
params_pydantic_model: type[BaseModel]
```
A Pydantic model that represents the function's parameters.

#### params_json_schema`instance-attribute`

```python
params_json_schema: dict[str, Any]
```
The JSON schema for the function's parameters, derived from the Pydantic model.

#### signature`instance-attribute`

```python
signature: Signature
```
The signature of the function.

#### takes_context`class-attribute``instance-attribute`

```python
takes_context: bool = False
```
Whether the function takes a RunContextWrapper argument (must be the first argument).

#### strict_json_schema`class-attribute``instance-attribute`

```python
strict_json_schema: bool = True
```
Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
as it increases the likelihood of correct JSON input.


#### to_call_args

```python
to_call_args(
    data: BaseModel,
) -> tuple[list[Any], dict[str, Any]]
```
Converts validated data from the Pydantic model into (args, kwargs), suitable for calling
the original function.


```python
def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
    """
    Converts validated data from the Pydantic model into (args, kwargs), suitable for calling
    the original function.
    """
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}
    seen_var_positional = False
    # Use enumerate() so we can skip the first parameter if it's context.
    for idx, (name, param) in enumerate(self.signature.parameters.items()):
        # If the function takes a RunContextWrapper and this is the first parameter, skip it.
        if self.takes_context and idx == 0:
            continue
        value = getattr(data, name, None)
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args and mark that *args is now seen
            positional_args.extend(value or [])
            seen_var_positional = True
        elif param.kind == param.VAR_KEYWORD:
            # e.g. **kwargs handling
            keyword_args.update(value or {})
        elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            # Before *args, add to positional args. After *args, add to keyword args.
            if not seen_var_positional:
                positional_args.append(value)
            else:
                keyword_args[name] = value
        else:
            # For KEYWORD_ONLY parameters, always use keyword args.
            keyword_args[name] = value
    return positional_args, keyword_args
```

---

### FuncDocumentation`dataclass`

Contains metadata about a python function, extracted from its docstring.

```python
@dataclass
class FuncDocumentation:
    """Contains metadata about a python function, extracted from its docstring."""
    name: str
    """The name of the function, via `__name__`."""
    description: str | None
    """The description of the function, derived from the docstring."""
    param_descriptions: dict[str, str] | None
    """The parameter descriptions of the function, derived from the docstring."""
```

#### name`instance-attribute`

```python
name: str
```
The name of the function, via `__name__`.

#### description`instance-attribute`

```python
description: str | None
```
The description of the function, derived from the docstring.

#### param_descriptions`instance-attribute`

```python
param_descriptions: dict[str, str] | None
```
The parameter descriptions of the function, derived from the docstring.

---

### generate_func_documentation

```python
generate_func_documentation(
    func: Callable[..., Any],
    style: DocstringStyle | None = None,
) -> FuncDocumentation
```
Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `func` | `Callable[..., Any]` | The function to extract documentation from. | _required_ |
| `style` | `DocstringStyle | None` | The style of the docstring to use for parsing. If not provided, we will attempt to<br>auto-detect the style. | `None` |

Returns:

| Type | Description |
| --- | --- |
| `FuncDocumentation` | A FuncDocumentation object containing the function's name, description, and parameter |
| `FuncDocumentation` | descriptions. |


```python
def generate_func_documentation(
    func: Callable[..., Any], style: DocstringStyle | None = None
) -> FuncDocumentation:
    """
    Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.
    Args:
        func: The function to extract documentation from.
        style: The style of the docstring to use for parsing. If not provided, we will attempt to
            auto-detect the style.
    Returns:
        A FuncDocumentation object containing the function's name, description, and parameter
        descriptions.
    """
    name = func.__name__
    doc = inspect.getdoc(func)
    if not doc:
        return FuncDocumentation(name=name, description=None, param_descriptions=None)
    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=style or _detect_docstring_style(doc))
        parsed = docstring.parse()
    description: str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
    )
    param_descriptions: dict[str, str] = {
        param.name: param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }
    return FuncDocumentation(
        name=func.__name__,
        description=description,
        param_descriptions=param_descriptions or None,
    )
```

---

### function_schema

```python
function_schema(
    func: Callable[..., Any],
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema
```
Given a python function, extracts a `FuncSchema` from it, capturing the name, description,
parameter descriptions, and other metadata.

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `func` | `Callable[..., Any]` | The function to extract the schema from. | _required_ |
| `docstring_style` | `DocstringStyle | None` | The style of the docstring to use for parsing. If not provided, we will<br>attempt to auto-detect the style. | `None` |
| `name_override` | `str | None` | If provided, use this name instead of the function's `__name__`. | `None` |
| `description_override` | `str | None` | If provided, use this description instead of the one derived from the<br>docstring. | `None` |
| `use_docstring_info` | `bool` | If True, uses the docstring to generate the description and parameter<br>descriptions. | `True` |
| `strict_json_schema` | `bool` | Whether the JSON schema is in strict mode. If True, we'll ensure that<br>the schema adheres to the "strict" standard the OpenAI API expects. We **strongly**<br>recommend setting this to True, as it increases the likelihood of the LLM providing<br>correct JSON input. | `True` |

Returns:

| Type | Description |
| --- | --- |
| `FuncSchema` | A `FuncSchema` object containing the function's name, description, parameter descriptions, |
| `FuncSchema` | and other metadata. |


```python
def function_schema(
    func: Callable[..., Any],
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """
    Given a python function, extracts a `FuncSchema` from it, capturing the name, description,
    parameter descriptions, and other metadata.
    Args:
        func: The function to extract the schema from.
        docstring_style: The style of the docstring to use for parsing. If not provided, we will
            attempt to auto-detect the style.
        name_override: If provided, use this name instead of the function's `__name__`.
        description_override: If provided, use this description instead of the one derived from the
            docstring.
        use_docstring_info: If True, uses the docstring to generate the description and parameter
            descriptions.
        strict_json_schema: Whether the JSON schema is in strict mode. If True, we'll ensure that
            the schema adheres to the "strict" standard the OpenAI API expects. We **strongly**
            recommend setting this to True, as it increases the likelihood of the LLM providing
            correct JSON input.
    Returns:
        A `FuncSchema` object containing the function's name, description, parameter descriptions,
        and other metadata.
    """
    # 1. Grab docstring info
    if use_docstring_info:
        doc_info = generate_func_documentation(func, docstring_style)
        param_descs = doc_info.param_descriptions or {}
    else:
        doc_info = None
        param_descs = {}
    # Ensure name_override takes precedence even if docstring info is disabled.
    func_name = name_override or (doc_info.name if doc_info else func.__name__)
    # 2. Inspect function signature and get type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = list(sig.parameters.items())
    takes_context = False
    filtered_params = []
    if params:
        first_name, first_param = params[0]
        # Prefer the evaluated type hint if available
        ann = type_hints.get(first_name, first_param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper or origin is ToolContext:
                takes_context = True  # Mark that the function takes context
            else:
                filtered_params.append((first_name, first_param))
        else:
            filtered_params.append((first_name, first_param))
    # For parameters other than the first, raise error if any use RunContextWrapper or ToolContext.
    for name, param in params[1:]:
        ann = type_hints.get(name, param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper or origin is ToolContext:
                raise UserError(
                    f"RunContextWrapper/ToolContext param found at non-first position in function"
                    f" {func.__name__}"
                )
        filtered_params.append((name, param))
    # We will collect field definitions for create_model as a dict:
    #   field_name -> (type_annotation, default_value_or_Field(...))
    fields: dict[str, Any] = {}
    for name, param in filtered_params:
        ann = type_hints.get(name, param.annotation)
        default = param.default
        # If there's no type hint, assume `Any`
        if ann == inspect._empty:
            ann = Any
        # If a docstring param description exists, use it
        field_description = param_descs.get(name, None)
        # Handle different parameter kinds
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args
            if get_origin(ann) is tuple:
                # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                args_of_tuple = get_args(ann)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    ann = list[args_of_tuple[0]]  # type: ignore
                else:
                    ann = list[Any]
            else:
                # If user wrote *args: int, treat as List[int]
                ann = list[ann]  # type: ignore
            # Default factory to empty list
            fields[name] = (
                ann,
                Field(default_factory=list, description=field_description),  # type: ignore
            )
        elif param.kind == param.VAR_KEYWORD:
            # **kwargs handling
            if get_origin(ann) is dict:
                # e.g. def foo(**kwargs: dict[str, int])
                dict_args = get_args(ann)
                if len(dict_args) == 2:
                    ann = dict[dict_args[0], dict_args[1]]  # type: ignore
                else:
                    ann = dict[str, Any]
            else:
                # e.g. def foo(**kwargs: int) -> Dict[str, int]
                ann = dict[str, ann]  # type: ignore
            fields[name] = (
                ann,
                Field(default_factory=dict, description=field_description),  # type: ignore
            )
        else:
            # Normal parameter
            if default == inspect._empty:
                # Required field
                fields[name] = (
                    ann,
                    Field(..., description=field_description),
                )
            else:
                # Parameter with a default value
                fields[name] = (
                    ann,
                    Field(default=default, description=field_description),
                )
    # 3. Dynamically build a Pydantic model
    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)
    # 4. Build JSON schema from that model
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)
    # 5. Return as a FuncSchema dataclass
    return FuncSchema(
        name=func_name,
        # Ensure description_override takes precedence even if docstring info is disabled.
        description=description_override or (doc_info.description if doc_info else None),
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        takes_context=takes_context,
        strict_json_schema=strict_json_schema,
    )
```