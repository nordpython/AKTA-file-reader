from numpy import shape
from numpy.core import shape_base
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from copy import copy
from pypage import pypage




def unicorn_ploty_graph(df,first="UV 1_280",second="Cond",third="pH",forth="Conc B"):
  uv_color = "#1f77b4"
  ph_color = "#2ca772"
  cond_color = "#f29d5f"
  concb_color = "#b5b5b5"

  UV = [c for c in df.columns if c[:2]=="UV"]

  axis_label = {
    UV[0]:f"UV {UV[0].split('_')[-1]}nm (mAU)",
    UV[1]:f"UV {UV[1].split('_')[-1]}nm (mAU)",
    UV[2]:f"UV {UV[2].split('_')[-1]}nm (mAU)",
    'Cond':"Conductivity (mS/cm)",
    'Conc B':"B Concentration (%)", 
    'pH':"pH",
    'System flow':"System flow (mL/min)",
    'Sample flow':"System flow (mL/min)",
    'PreC pressure':"PreC pressure (MPa)",
    'System pressure':"System pressure (MPa)",
    'Sample pressure':"Sample pressure (MPa)",
  }



  fig = go.Figure()

  fig.add_trace(go.Scatter(
      x=df["mL"],
      y=df[first],
      yaxis="y",
      line=dict(
              color=uv_color
          ),
      fill = "tozeroy",
      name=axis_label[first]
  ))
  fig.update_layout(
      xaxis=dict(
          domain=[0.05, 0.85],
          title="mL",
      ),
      yaxis=dict(
          title=axis_label[first],
          titlefont=dict(
              color=uv_color
          ),
          tickfont=dict(
              color=uv_color
          )
          )
      )
  
  if second:
    fig.add_trace(go.Scatter(
        x=df["mL"],
        y=df[second],
        yaxis="y2",
        line=dict(
                color=cond_color
            ),
        name=axis_label[second]
    ))

    fig.update_layout(
            yaxis2=dict(
          title=axis_label[second],
          titlefont=dict(
              color=cond_color
          ),
          tickfont=dict(
              color=cond_color
          ),
          anchor="x",
          side="right",
          overlaying="y")
    )
    
  
  if third:
    fig.add_trace(go.Scatter(
        x=df["mL"],
        y=df[third],
        yaxis="y3",
        line=dict(
                color=ph_color
            ),
        name=axis_label[third]
    ))

    fig.update_layout(
          yaxis3=dict(
          title=axis_label[third],
          titlefont=dict(
              color=ph_color
          ),
          tickfont=dict(
              color=ph_color
          ),
          anchor="free",
          side="right",
          #range=(2,12),
          overlaying="y", autoshift=True)
    )
  
  if forth: 
    fig.add_trace(go.Scatter(
        x=df["mL"],
        y=df[forth],
        yaxis="y4",
        name=axis_label[forth],
        line=dict(
                color=concb_color
            ),
    ))

    fig.update_layout(
          yaxis4=dict(
          title=axis_label[forth],
          titlefont=dict(
              color=concb_color,
          ),
          tickfont=dict(
              color=concb_color
          ),
          anchor="free",
          side="right",
          overlaying="y", autoshift=True),
    )



  fig.update_layout(
      template="plotly_white",
      plot_bgcolor='rgba(0,0,0,0)',
      font=dict(
        size=14,
      ),
      #title=dict(text='Chromatogram',
      #           font=dict(size=20),
      #            x=0.2,
      #            xanchor='center'
      #          ),
      width=900,
      height=540,
      legend=dict(
          yanchor="bottom",
          y=1,
          xanchor="left",
          x=0,
          font=dict(size=12),
      ))



  
  
  return fig



def annotate_fraction(fig,frac_df,phase=None,rectangle=True,text=True,palette=None,annotations=None):

  fig =copy(fig)
  
  if not palette:
    palette = sns.color_palette("Blues", len(frac_df))
  
  use_color_palette = {}

  shapes = []
  texts = []
  phase_shapes = []
  phase_texts = []

  for i,(index, row) in enumerate(frac_df.iterrows()):
    if annotations:
      if not row["Fraction_Start"] in annotations:
        continue

    color = f"rgb({int(palette[i][0]*255)},{int(palette[i][1]*255)},{int(palette[i][2]*255)})"
    use_color_palette[row["Fraction_Start"]] = palette[i]

    if rectangle:
      
      shapes.append(dict(type="rect",
                    x0=row["Start_mL"], y0=0, x1=row["End_mL"], y1=row["Max_UV"],
                    line=dict(color=color,width=2),
                    ))

    if text:
      texts.append(dict(
                        x=(row["Start_mL"]+row["End_mL"])/2,
                        y=0,
                        xref="x",
                        yref="y",
                        text=row["Fraction_Start"],
                        align='center',
                        showarrow=False,
                        yanchor='top',
                        textangle=90,
                        font=dict(
                        size=10
                        ),
                        bgcolor=color,

                        opacity=0.8))


  if phase is not None:
    palette_phase = sns.color_palette(n_colors=len(phase))

   
    max_mL = frac_df["Max_UV"].max()*1.1
    
    for i,row in phase.iterrows():
      color = f"rgb({int(palette_phase[i][0]*255)},{int(palette_phase[i][1]*255)},{int(palette_phase[i][2]*255)})"
      phase_shapes.append(dict(type="rect",
                      x0=row["Start_mL"], y0=0, x1=row["End_mL"], y1=max_mL,
                      layer="below",
                      line=dict(color=color,width=0),
                      fillcolor=color,
                      opacity=0.1
                      ))
      
      if "Phase" in phase.columns:
          phase_texts.append(dict(
                        x=(row["Start_mL"]+row["End_mL"])/2,
                        y=max_mL,
                        xref="x",
                        yref="y",
                        text=row["Phase"],
                        align='center',
                        showarrow=False,
                        yanchor='top',
                        font=dict(
                        size=12
                        ),
                        opacity=1))


  # shapesとannotationsを追加
  fig.update_layout(
      shapes=shapes+phase_shapes,
      annotations=texts+phase_texts
  )                  
  fig.update_shapes(dict(xref='x', yref='y'))

  fig.update_layout(
    width=900,
    height=600,
    updatemenus=[
      dict(
          type="buttons",
          direction="down",
          yanchor="bottom",
          y=1.05,
          xanchor="right",
          x=0.85,
          showactive=True,
          active=0,
          font=dict(size=12),
          buttons=[
              dict(
                  args=[{f"shapes[{k}].visible": True for k in range(len(shapes))}],
                  args2=[{f"shapes[{k}].visible": False for k in range(len(shapes))}],
                  label="fraction box",
                  method="relayout"
              ),
          ]
      ),
      dict(
          type="buttons",
          direction="down",
          yanchor="bottom",
          y=1.05,
          xanchor="right",
          x=1,
          showactive=True,
          active=0,
          font=dict(size=12),
          buttons=[
              dict(
                  args=[{f"annotations[{k}].visible": True for k in range(len(texts))}],
                  args2=[{f"annotations[{k}].visible": False for k in range(len(texts))}],
                  label="fraction text",
                  method="relayout"
              ),
          ]
      ),
      dict(
          type="buttons",
          direction="down",
          yanchor="bottom",
          y=1.15,
          xanchor="right",
          x=0.85,
          showactive=True,
          active=0,
          font=dict(size=12),
          buttons=[
              dict(
                  args=[{f"shapes[{k}].visible": True for k in range(len(shapes),len(shapes+phase_shapes))}],
                  args2=[{f"shapes[{k}].visible": False for k in range(len(shapes),len(shapes+phase_shapes))}],
                  label="phase box",
                  method="relayout"
              ),
          ]
      ),
         dict(
          type="buttons",
          direction="down",
          yanchor="bottom",
          y=1.15,
          xanchor="right",
          x=1,
          showactive=True,
          active=0,
          font=dict(size=12),
          buttons=[
              dict(
                  args=[{f"annotations[{k}].visible": True for k in range(len(texts),len(shapes+phase_texts))}],
                  args2=[{f"annotations[{k}].visible": False for k in range(len(texts),len(shapes+phase_texts))}],
                  label="phase text",
                  method="relayout"
              ),
          ]
      )
  ],
  )
  return fig,use_color_palette



def annotate_page(image, lanes, lane_width=30,rectangle=True,text=True,palette_dict=None,annotations=None):

  fig = px.imshow(image)
  height, width = image.shape[:2]
  fig.update_layout(
      template="plotly_white",
      title=dict(text='CBB stain',
                 font=dict(size=24),
                  x=0.5,
                  y=0.95,
                  xanchor='center',
                  #yanchor="bottom"
                ),
      width=width,
      height=height
  )  

  fig.update_layout(
    # プロットの背景を透明に
    plot_bgcolor='rgba(0,0,0,0)',
    # 図全体の背景を透明に（必要に応じて）
    paper_bgcolor='rgba(0,0,0,0)',
    # x軸の線を消す
    xaxis=dict(
        showline=False,
        showgrid=False,
        zeroline=False,
    ),
    # y軸の線を消す
    yaxis=dict(
        showline=False,
        showgrid=False,
        zeroline=False,
    )
  )

  if not annotations:
      annotations = list(range(len(lanes)))

  if not palette_dict:
      palette = sns.color_palette("Set1", len(lanes))
      #annotations = list(range(len(lanes)))
      palette_dict = {a:p for a,p in zip(annotations,palette)}


  shapes = []
  texts = []
  i=0
  for label,lane in zip(annotations,lanes):
    if not label in palette_dict.keys():
        continue
    
    if label == "":
      continue

    color = f"rgb({int(palette_dict[label][0]*255)},{int(palette_dict[label][1]*255)},{int(palette_dict[label][2]*255)})"

    if rectangle:
      lane_coord = pypage.get_lane(image,lane,lane_width=50)
      
      shapes.append(dict(type="rect",
                    x0=lane_coord.x0, y0=lane_coord.y0, x1=lane_coord.x1, y1=lane_coord.y1,
                    line=dict(color=color,width=2),
                    ))

    if text:
      texts.append(dict(
                            x=lane, y=100,
                            xref="x",
                            yref="y",
                            text=f"{label}",
                            align='center',
                            showarrow=False,
                            yanchor='bottom',
                            textangle=90,
                            font=dict(
                            size=18,
                            ),
                            bgcolor=color,
                            opacity=0.8))

                            
  fig.update_layout(coloraxis_showscale=False)
  fig.update_xaxes(showticklabels=False)
  fig.update_yaxes(showticklabels=False)

    # shapesとannotationsを追加
  fig.update_layout(
      shapes=shapes,
      annotations=texts
  )                  
  fig.update_shapes(dict(xref='x', yref='y'))

  fig.update_layout(
  updatemenus=[
      dict(
          type="buttons",
          direction="down",
          x=1.1,
          y=1.1,
          showactive=True,
          active=0,
          font=dict(size=12),
          buttons=[
              dict(
                  args=[{f"shapes[{k}].visible": True for k in range(len(shapes))}],
                  args2=[{f"shapes[{k}].visible": False for k in range(len(shapes))}],
                  label="Rectangle ☑",
                  method="relayout"
              ),
          ]
      ),
      dict(
          type="buttons",
          direction="down",
          x=1.1,
          y=1.2,
          showactive=True,
          active=0,
          font=dict(size=12),
          buttons=[
              dict(
                  args=[{f"annotations[{k}].visible": True for k in range(len(texts))}],
                  args2=[{f"annotations[{k}].visible": False for k in range(len(texts))}],
                  label="Annotation ☑",
                  method="relayout"
              ),
          ]
      )
  ],
  )

  return fig