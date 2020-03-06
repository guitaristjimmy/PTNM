# Project PTNM
## Personal Taste Navigation Model
#### Team : Find a Secret Key Dongguk Univ.

### Overview ::
It is music recommendation program using 8 feature vectors extracted from music
Recently, hybrid recommendation systems are widely used to recommend content using a combination of Collaborative Filtering (CF) and Item Based Recommendation. 
For example, "Spotify" provides hybrid recommendation service using three models, and is provided as an Open API. 
 This is the recommended way to do Spotify.
- Collaborative Filtering Model
- NLP through tags that characterize the music source
- Classification of new songs via CNN

It is true that these models provide a sufficiently high level of satisfaction. However, as time goes by, users become less satisfied with receiving only similar music recommendations.
Therefore, we aimed to create a model that would suit users' tastes and recommend new music that I've never experienced before.
PTNM started with an idea from CNN. CNN is the extraction of frequency features through the Convolution Layer and classification or prediction through the Neural Network.
PTNM used vectors that extracted eight features instead of the Convolution Layer, and for ways to reflect user evaluations of frequency features on behalf of the Neural Network, see Page Rank Algorithm. Based on user evaluation, weights are obtained for frequency characteristics and then re-recommended by ranking them.

Please refer to the report for more information.