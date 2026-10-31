"""Deterministic figures from frozen table rows only; no statistical computation."""
import io,os,tempfile
from pathlib import Path
from . import analysis_schema as wire,data_artifacts as data

def render_figures(root,tables,table_records,provenance):
    with tempfile.TemporaryDirectory(prefix="star_v2_mpl_") as temp:
        old=os.environ.get("MPLCONFIGDIR");os.environ["MPLCONFIGDIR"]=temp
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            specifications={
                "pair_ate":("pooled_rmse","Pair ATE pooled RMSE","outcome"),
                "comparisons":("paired_rmse_difference","Paired RMSE differences (pointwise 95%)","outcome"),
                "j_coverage":("coverage_including_failures","J coverage including failed intervals","proportion"),
                "node_error":("pooled_rmse","Node response / ITE pooled RMSE","outcome"),
                "group_ate":("pooled_rmse","Center / leaf / all group ATE RMSE","outcome"),
                "center_leaf":("pooled_rmse","Center-minus-leaf difference RMSE","outcome"),
                "support_strata":("pooled_rmse","Natural Support strata (descriptive only)","outcome"),
                "runtime":("mean","Mean summed Pair runtime per mechanism","seconds"),
                "oracle_context":("pooled_rmse","Oracle-context reference: Pair ATE RMSE","outcome"),
                "oracle_pair":("pooled_rmse","Oracle-pair reference: Pair ATE RMSE","outcome")}
            receipts=[]
            with matplotlib.rc_context({"svg.hashsalt":"star-v2-frozen-analysis-v1",
                    "font.family":"DejaVu Sans","font.size":8}):
                for name,(metric,title,unit) in specifications.items():
                    rows=[r for r in tables[name] if r["metric"]==metric]
                    if name in ("oracle_context","oracle_pair"):
                        rows=[r for r in rows if r["source_selector"].get("family")=="pair_ate"]
                    if name=="support_strata":
                        rows=[r for r in rows if r["group"]=="all" and r["center_t"] is None]
                    fig=Figure(figsize=(12,max(3,.27*len(rows)+1.6)),layout="constrained")
                    FigureCanvasAgg(fig);ax=fig.add_subplot(111)
                    labels=[]
                    for i,row in enumerate(rows):
                        label=row["method_id"]+" | "+row["effect"]
                        if row["group"]!="all":label+=" | "+row["group"]
                        if row["center_t"] is not None:label+=" | center_t="+str(row["center_t"])
                        if row["support_reason"]:label+=" | "+row["support_reason"]
                        labels.append(label)
                        if row["value"] is None:
                            ax.text(.01,i,"N/A: "+row["status"],transform=ax.get_yaxis_transform(),va="center",color="#777777")
                        elif row["ci_lower"] is not None:
                            ax.plot([row["ci_lower"],row["ci_upper"]],[i,i],color="#0072B2",linewidth=1.5)
                            ax.plot(row["value"],i,"o",color="#0072B2",markersize=4)
                        else:ax.plot(row["value"],i,"o",color="#0072B2",markersize=4)
                    ax.set_yticks(range(len(rows)),labels)
                    ax.invert_yaxis();ax.set_xlabel(unit);ax.set_title(title)
                    ax.grid(axis="x",alpha=.2)
                    if name=="j_coverage":
                        ax.axvline(.90,color="#D55E00",linestyle="--",label="Frozen overall target .90")
                        ax.legend()
                    if name=="comparisons":ax.axvline(0,color="#777777",linewidth=.7)
                    for extension in ("svg","png"):
                        output=io.BytesIO()
                        meta={"Date":None,"Creator":"star_v2 frozen analysis"} if extension=="svg" else {"Software":"star_v2 frozen analysis"}
                        fig.savefig(output,format=extension,dpi=120,metadata=meta)
                        receipts.append(data._write(root,"figures/"+name+"."+extension,output.getvalue()))
                    receipts.append(wire.write_json(root,"figures/"+name+".metadata.json",{
                        "schema_version":1,"figure":name,"unit":unit,"methods":sorted({r["method_id"] for r in rows}),
                        "effects":sorted({r["effect"] for r in rows}),"source_table":table_records[name],
                        "source_row_ids":[r["row_id"] for r in rows],"rows":rows,
                        "denominator_and_failures":"Per selected row n_requested/n_valid/n_failed_or_missing",
                        "statistical_intervals":"Only comparison/J coverage rows have pointwise 95% CI; other intervals NOT_REQUESTED",
                        "input_provenance":provenance,"scientific_performance_pass_claimed":False}))
                    fig.clear()
            return receipts
        finally:
            if old is None:os.environ.pop("MPLCONFIGDIR",None)
            else:os.environ["MPLCONFIGDIR"]=old
