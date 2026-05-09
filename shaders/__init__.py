from importlib.resources import files, as_file

TEXT_EXTENSIONS = {".vert", ".frag"}

shader_dir = files("bereshit.shaders")

with as_file(shader_dir) as real_shader_dir:

    for file in real_shader_dir.iterdir():

        if not file.is_file():
            continue

        if file.suffix in TEXT_EXTENSIONS:

            source = file.read_text(encoding="utf-8")

        else:

            data = file.read_bytes()
## need to add
##[tool.setuptools.package-data]
##"bereshit.shaders" = ["*"]