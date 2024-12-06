### Dataset -- HateMM
|Dataset| Language| # Hate| # Non-Hate| Resource |
|:-:|:-:|:-:|:-:| :-:|
|HateMM|English|431|652| BitChute| 



### Dataset -- MultiHateClip
| Dataset       | Language | # Hateful | # Offensive | # Normal | Resource  |
|:-------------:|:--------:|:------:|:-----------:|:--------:|:---------:|
| MultiHateClip | English  |    82    |      256       |     662     | YouTube   |
| MultiHateClip | Chinese  |    128    |      194       |     678     | Bilibili  |



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
    <td>VideoLLaMA2 (30Frames)</td>
    <td>0.62442</td>
    <td>0.54811</td>
    <td>0.30536</td>
    <td>0.39221</td>
  </tr>
    <tr>
    <td>VideoLLaMA2-AV</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>  
  </tr>
  <tr>
    <td>LLaVA-Next-Video(Image-Text, 24Frames)</td>
    <td>0.55863</td>
    <td>0.46252</td>
    <td>0.67285</td>
    <td>0.54820</td>
  </tr>
  <tr>
    <td>LLaVA-OneVision(Image-Text. 24Frames)</td>
    <td>0.65836</td>
    <td>0.80198</td>
    <td>0.18794</td>
    <td>0.30451</td>
  </tr>
</table>
