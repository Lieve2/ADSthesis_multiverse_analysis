from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from PIL import Image

app = Dash(__name__, external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME])

uu_img = Image.open("/Users/lievegobbels/PycharmProjects/ADSthesis/figures/uulogo.png") #"https://user-images.githubusercontent.com/72614349/185497519-733bdfc3-5731-4419-9a68-44c1cad04a78.png"
github = "fa-brands fa-github"
linkedin = "fa-brands fa-linkedin"
dash_url = "https://dash.plotly.com/"
git_lieve = "https://github.com/Lieve2"
git_project = "https://github.com/Lieve2/ADSthesis_multiverse_analysis.git"
lkdin_lieve = "https://www.linkedin.com/in/lievegobbels"



def make_link(text, icon, link):
    return html.Span(html.A([html.I(className=icon + " ps-2"), text], href=link))

button = dbc.Button(
    "Go to project code", color="secondary", href=git_project, size="sm", className="mt-2 ms-1"
)

cover_img = html.A(
    dbc.CardImg(
        src=uu_img,
        className="img-fluid rounded-start",
    ),
    href=git_project,
)

text = dcc.Markdown(
    "This dashboard is made with Plotly [Dash]({dash_url}). "
    f"Click the button for more details.",
    className="ps-2",
)

authors = html.P(
    [
        "By ",
        make_link("", linkedin, lkdin_lieve),
        make_link(" Lieve Göbbels", github, git_lieve),
    ],
    className="card-text p-2",
)

card = dbc.Card(
    [
        dbc.Stack(
            [
                dbc.Col(
                    cover_img,
                    width=2
                ),
                dbc.Col(
                    text,
                    width=10,
                    align="center"
                ),
                dbc.Col(
                    button,
                    width=4,
                    align="center"
                ),
            ],
            className="g-0 d-flex align-items-center text-center",
            gap=2.5,
        ),
        dbc.Row(dbc.Col(authors),
        justify="center"),
    ],
    className="my-5 small",
    style={
        "maxWidth": "36em",
    "maxHeight": "18em",
    "padding-top": "0.5em",
    "padding-bottom": "0.5em"
    },
)

app.layout = dbc.Container(card, fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True)