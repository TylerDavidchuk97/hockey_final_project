# hockey_final_project
Final project for Concordia CBDS-5
Project Outline

⦁Gather Hockey statistics of players from the history of the nhl using the Elite Prospects Api and do the following on the dataset


⦁First off I take the career of the player and find out given all players that played who their closest compareable is. This would be done using all of their career stats. E.g.(Sidney Crosby's goal totals, assist totals, game winning goals, points per game, plus minus, penalty minutes. Who would be his closest compareable player maybe Steve Yzerman). THis is done using a KNN

⦁All career stats taken and used for the KNN I used the mean of all of their career stats except for the games played and total career points which are summmed

⦁Given any player that is searched for I wanted to find out the following about them. Given their career so far or even retired players I created an LSTM model that would give a prediction about what their points per game next season would be, I only included players with at least 3 career seasons in the NHL without a games player requirement, the LSTM would predict any next season for any player that meets the 3 seasons of eligibility. E.g.(Sidney Crosby ppg for the 2022-2023 season)


⦁All career stats used for eligible players for the LSTM points per game prediction were divided by their games played for that season. This is fair as players a long time ago near the start of the league didn't play near as many games as players do today (82 games a season as of the 2021-2022 NHL season).


⦁This project takes user input via browser using html and displays the results as required for the deployment portion specified by the project guidelines.


⦁The html deployment was accomplished using flask 


