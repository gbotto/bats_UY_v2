source("multi_rf1_r.R")
# library(tidyverse)
library(randomForest)
library(ggplot2)
library(labeling)
shinyApp(
  ui = shinyUI(fluidPage(
    titlePanel("Bat's ultrasound Classifier" ),
    
    sidebarLayout(
      sidebarPanel(
        
        
        fileInput('file1', 'Upload TXT Sonobat file',
                  accept=c('text', 'text', '.txt')),
        tags$hr()),
      mainPanel(
        plotOutput("SP_Plot"),
        sidebarPanel(
          downloadButton('downloadData', 'Download'))
      )),
    fluidRow(column(4, includeHTML("textoweb.html")))
  )
),


  server = (function(input, output) {
    
    bats_UY <- function(mod, db){
      prob.matrix <- predict(mod, newdata = db, type = "prob")
      spp <- colnames(prob.matrix)
      result <- data.frame(Pred.SP = spp[apply(prob.matrix, 1, which.max)],
                           Prob.SP = ifelse(apply(prob.matrix, 1, max)>=0.43, apply(prob.matrix, 1, max), NA), stringsAsFactors = FALSE)
      result$Pred.SP <- ifelse(is.na(result$Prob.SP), "Unknown", result$Pred.SP)
      return(result)
    }
   

    
    
    
    output$SP_Plot <- renderPlot({
      inFile <- input$file1
      if (is.null(inFile))
        return(NULL)
      entrada <- read.delim2(inFile$datapath)
      mod1 <- multi_rf1
      umbral <- input$umbral
      salida <<- data.frame(bats_UY(mod1, entrada))
      ggplot(data = salida, aes(Pred.SP, fill=Pred.SP))+
        geom_bar()+
        ylab("Absolut Frequency")+
        ggtitle(input$modelo)+
        theme(axis.title.x=element_blank(),
              axis.text.x = element_text(angle=30, hjust=1, vjust=1)) 
    }) 
    output$downloadData <- downloadHandler(
      filename = function() { paste(Sys.Date(),Sys.time(), '.csv', sep='') },
      content = function(file) {
        write.csv2(salida, file)
      })
  })

  )
