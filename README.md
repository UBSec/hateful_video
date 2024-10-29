### Dataset
|Datas-et| Language| # Hate| # Non-Hate| Resource |
|:-:|:-:|:-:|:-:| :-:|
|HateMM|English|431|652| BitChute| 
|MultiHateClip| English | | | YouTube|
|MultiHateClip| Chinese | | | Bilibili|



### Task 1: General Prompt

``` "Is there any hateful content in this video? Respond 'Yes' or 'No' and explain why."```

<table>
  <tr>
    <th>MLLMs</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <th colspan="5" style="text-align:center"><em>Closed-source</em></th>
  </tr>
  <tr>
    <td>Gemini-1.5-pro</td>
    <td>0.64380</td>
    <td>0.42741</td>
    <td>0.94642</td>
    <td>0.58889</td>
  </tr>
  <tr>
    <td>Azure AI Video Indexer</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th colspan="5" style="text-align:center"><em>Open-source</em></th>
  </tr>
  <tr>
    <td>VideoChat2</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>VideoLLaMA2 (30FPS)</td>
    <td>0.62442</td>
    <td>0.54811</td>
    <td>0.30536</td>
    <td>0.39221</td>
  </tr>
  <tr>
    <td>LLaVA-Next-Video</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>LLaVA-OneVision</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>
