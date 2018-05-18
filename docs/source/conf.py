# -*- coding: utf-8 -*-
#
# Minerva documentation build configuration file, created by
#
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
from collections import OrderedDict
from configparser import ConfigParser

# Get the root package path
repo_root = os.path.abspath('../..')
src_root = os.path.join(repo_root, 'src/minerva_lib')
all_paths = [d for d, n, f in os.walk(src_root)]
# Add all package paths and root path to sys.path
map(sys.path.append, all_paths+[repo_root, '.'])


def read_version(root):
    ''' Version string of this module '''
    config = ConfigParser()
    config.read(os.path.join(root, 'setup.cfg'))
    return config.get('metadata', 'version')

# Short and long version are same
version = read_version(repo_root)
release = read_version(repo_root)

# General information about the project.
github_repo = 'minerva-lib-python'
author = 'Douglas Russell'
github_user = 'sorgerlab'
project = github_repo
copyright = '2018'
language = 'en'

# -- General configuration ------------------------------------------------

needs_sphinx = '1.5.3'

# Add any Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

# Autodoc configuration
autodoc_member_order = 'groupwise'
templates_path = ['_templates']
todo_include_todos = False
exclude_patterns = []
source_suffix = '.rst'
primary_domain = 'py'

# Syntax highlighting
pygments_style = 'monokai'
# The master toctree document.
master_doc = 'content'


def edit_docstring(app, what, name, obj, options, lines):
    ''' Standard hook for custom docstrings
    http://www.sphinx-doc.org/en/master/ext/autodoc.html
    #event-autodoc-process-docstring
    '''
    if lines and what == 'data':
        del lines[:]


def setup(app):
    ''' called once autodoc loaded '''
    app.connect('autodoc-process-docstring', edit_docstring)


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'alabaster'
html_static_path = ['_static']

# Add links on the sidebar
extra_links = OrderedDict()
extra_links['General Index'] = 'genindex.html'
extra_links['Module Index'] = 'py-modindex.html'
extra_links['Github Wiki'] = ('https://github.com/'
                                '{}/{}/wiki'.format(github_user, github_repo))

# Specific options for Alabaster theme
html_theme_options = dict(
    logo='minerva.png',
    logo_name=False,
    font_size='1.0em',
    page_width='875px',
    fixed_sidebar=True,
    github_button=False,
    sidebar_collapse=True,
    show_powered_by=False,
    github_user=github_user,
    github_repo=github_user,
    code_font_size='inherit',
    extra_nav_links=extra_links,
    description='The Minerva Library',
    code_font_family='"Anonymous Pro", monospace',
    sidebar_link_underscore='#ACE',
    link='#ACE',
    gray_1='#CCC',
    gray_2='#CCC',
    gray_3='#CCC',
    pre_bg='#000',
    body_text='#CCC',
    footer_text='#CCC',
    sidebar_link='#CCC',
    link_hover='#FFF',
    sidebar_header='#FFF',
    anchor_hover_fg='#FFF',
    anchor_hover_bg='#000',
    font_family="'Lato', Arial, sans-serif",
    head_font_family="'Raleway', Arial, sans-serif"
)

html_sidebars = {
    '**': [
        'about.html',
        'searchbox.html',
        'relations.html',
        'navigation.html',
    ]
}


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'minerva', 'Minerva-Lib /docs', [author], 1)
]
